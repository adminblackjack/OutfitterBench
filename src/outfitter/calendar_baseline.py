"""Run the paper's fixed Calendar policy through public tools, without an LLM."""
from __future__ import annotations

from dataclasses import replace
import json
from pathlib import Path

from .config import OutfitterConfig
from .paper_baselines import PaperCalendar, PROTOCOL
from .recovery_state import ENVIRONMENT_PROTOCOL
from .run_scripted_merchant import DailyJournal, NoModel
from .runner import OutfitterRunner, _implementation_identity
from .simple_policy_stress import READ_TOOLS
from .simple_policy_stress_run import PublicRPC, RoundLimit


class BusinessTerminated(Exception):
    """A valid tool action has terminated the business."""


class CalendarRPC(PublicRPC):
    """Persist actual precommitted batches, matching the paper's interaction budget."""
    def _record_round(self, queries):
        message = {'role': 'assistant', 'content': None, 'tool_calls': [
            {'id': f'simple-d{self.state.day:03d}-{self.serial+i+1:05d}', 'type': 'function',
             'function': {'name': name, 'arguments': json.dumps(args, ensure_ascii=False)}}
            for i, (name, args) in enumerate(queries)]}
        self.journal.db.execute('INSERT INTO messages(day,scope,value) VALUES(?,?,?)',
                                (self.state.day, 'controller', json.dumps(message, ensure_ascii=False)))

    def __call__(self, name, args):
        self._begin(closing=name == 'close_business_day')
        self._record_round([(name, args)])
        result = self._execute(name, args)
        if self.state.done and name != 'close_business_day':
            raise BusinessTerminated('business terminated during a manager action')
        return result

    def batch(self, queries):
        queries = json.loads(json.dumps(queries))
        if any(name not in READ_TOOLS for name, _ in queries):
            raise ValueError('only independent read-only batches are allowed')
        result = []
        for start in range(0, len(queries), self.batch_size):
            chunk = queries[start:start+self.batch_size]
            self._begin()
            self._record_round(chunk)
            result.extend(self._execute(name, args) for name, args in chunk)
        return result


def _check_protocol(config):
    config.validate()
    if (config.max_days, config.runoff_days, config.max_iterations_per_day,
            config.max_read_only_batch_size, config.assortment_mode) != (200, 14, 200, 5, 'rental_only'):
        raise ValueError('Calendar uses the paper protocol: rental_only, 200+14 days, 200 rounds, batches of 5')
    from .data import OutfitterDataset
    if OutfitterDataset.load(config.data_dir).metadata['environment_protocol'] != ENVIRONMENT_PROTOCOL:
        raise ValueError('Calendar requires the V54 environment data')


def run_calendar(config=None, *, run_id=None, resume_dir=None, stop_after_day=None):
    """Stop at daily boundaries; resumption uses the frozen run configuration."""
    if stop_after_day is not None and not 1 <= stop_after_day <= 214:
        raise ValueError('--stop-after-day must be between 1 and 214')
    if resume_dir is not None:
        if config is not None or run_id is not None:
            raise ValueError('resumption cannot override the saved configuration or run ID')
        directory = Path(resume_dir).expanduser().resolve()
        protocol = json.loads((directory / 'calendar_protocol.json').read_text())
        if protocol['policy'] != 'Calendar' or protocol['protocol'] != PROTOCOL:
            raise ValueError('not a matching Calendar run')
        manifest = json.loads((directory / 'manifest.json').read_text())
        values = dict(manifest['config'], data_dir=str(directory / manifest['runtime_data']))
        config = OutfitterConfig.from_dict(values)
        _check_protocol(config)
        runner = OutfitterRunner.resume(run_dir=directory, provider=NoModel())
    else:
        if config is None:
            raise ValueError('a configuration is required for a new Calendar run')
        _check_protocol(config)
        config = replace(config, model='rule-calendar', base_url=None, reasoning_effort=None,
                         thinking_type=None, api_key_env='OUTFITTER_UNUSED_RULE_KEY',
                         provider_extra_body={}, charge_model_usage_to_cash=False)
        runner = OutfitterRunner.create(config=config, provider=NoModel(), run_id=run_id)
        directory = runner.store.run_dir
        protocol = {'policy': 'Calendar', 'mode': 'C', 'protocol': PROTOCOL,
                    'information': 'public tool results only', 'no_llm': True,
                    'prices': PaperCalendar('C', None).prices, 'price_scale': 1.25,
                    'extra_return_buffer_days': 0, 'horizon': [200, 14],
                    'core_identity': _implementation_identity(),
                    'max_daily_interaction_rounds': 200, 'max_read_only_batch_size': 5}
        (directory / 'calendar_protocol.json').write_text(json.dumps(protocol, indent=2) + '\n')

    state = runner.store.load_state()
    journal = DailyJournal(runner.store, runner.tools)
    journal.db.execute('CREATE TABLE IF NOT EXISTS calendar_decisions (day INTEGER PRIMARY KEY, value TEXT NOT NULL)')
    try:
        while not state.done and (stop_after_day is None or state.day <= stop_after_day):
            day = state.day
            journal.begin()
            if day > config.max_days:
                reply = journal.execute(state, f'runoff-{day}', 'close_business_day', {})
                if not reply.ok:
                    raise RuntimeError(reply.data)
                journal.commit(state)
                continue
            journal.opening(state, {'role': 'user', 'content': f'Day {day}: public-tool rule policy C.'})
            rpc = CalendarRPC(journal, state, runner.tools, 200, 5)
            merchant = PaperCalendar('C', rpc)
            try:
                merchant.day()
            except RoundLimit:
                merchant.log('budget_exhausted')
            except BusinessTerminated:
                if not state.bankrupt:
                    raise
            reply = rpc('close_business_day', {}) if not state.done else {'ok': True}
            if (rpc.errors or not reply['ok']) and not state.bankrupt:
                raise RuntimeError({'day': day, 'errors': rpc.errors, 'close': reply})
            journal.db.execute('INSERT INTO calendar_decisions(day,value) VALUES(?,?)',
                               (day, json.dumps(merchant.decisions)))
            journal.commit(state)
        summary = runner.summary(state, 'completed' if state.done else 'engineering_stop')
        if abs(summary['cash_flow']['reconciliation_difference']) >= .005:
            raise RuntimeError('cash-flow reconciliation failed')
        if journal.db.execute('SELECT count(*) FROM model_calls').fetchone()[0]:
            raise RuntimeError('Calendar must not make model calls')
        counts = dict(journal.db.execute('SELECT name,count(*) FROM tool_calls GROUP BY name'))
        summary['calendar'] = {'protocol': PROTOCOL, 'model_calls': 0, 'tool_counts': counts,
                               'completed_days': len(state.daily_history)}
        runner.store.write_summary(summary)
        return summary
    finally:
        journal.close()
