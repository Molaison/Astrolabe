"""Atomic native-call quota shared by isolated execution owners.

Policy scopes and the owning backend define the charge unit. Reserved or issued
attempts survive failure and are never refunded because a collector timed out.
"""
import fcntl
import json
from pathlib import Path


def _update(path,operation):
    from .io import atomic_json
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    with path.with_suffix('.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        data=json.loads(path.read_text()) if path.exists() else dict(entries={})
        result=operation(data)
        atomic_json(path,data)
        return result


def reserve_calls(policy,tasks):
    """Reserve before dispatch; uncertain reservations are never silently refunded."""
    scope=str(policy['scope'])
    def operation(data):
        contract=dict(max_total=int(policy['max_total']),max_per_case=int(policy['max_per_case']))
        if 'contract' in data and data['contract']!=contract:
            raise ValueError('Native call quota changed')
        data['contract']=contract
        admitted=[];refused=[]
        for task in tasks:
            if task['kind']=='rescore':
                admitted.append(task);continue
            token=f"{scope}:{task['id']}:{task.get('attempt',1)}"
            existing=data['entries'].get(token)
            if existing is None:
                scope_count=sum(e['scope']==scope for e in data['entries'].values())
                if len(data['entries'])>=contract['max_total'] or scope_count>=contract['max_per_case']:
                    refused.append(task);continue
                existing=dict(scope=scope,candidate_id=task['id'],attempt=task.get('attempt',1),
                              seed=task['seed'],status='reserved')
                data['entries'][token]=existing
            expected=(scope,task['id'],task.get('attempt',1),task['seed'])
            if tuple(existing[k] for k in ('scope','candidate_id','attempt','seed'))!=expected:
                raise ValueError('Native reservation identity mismatch')
            if existing['status']!='reserved':
                raise RuntimeError('An issued Native attempt requires native reconciliation or explicit new retry')
            task[str(policy.get('token_field', 'charge_token'))]=token
            admitted.append(task)
        return admitted,refused
    return _update(policy['ledger'],operation)


def claim_call(policy,task):
    def operation(data):
        entry=data['entries'][task[str(policy.get('token_field', 'charge_token'))]]
        expected=(str(policy['scope']),task['id'],task.get('attempt',1),task['seed'])
        if tuple(entry[k] for k in ('scope','candidate_id','attempt','seed'))!=expected or entry['status']!='reserved':
            raise RuntimeError('Native request token is invalid or already issued')
        entry['status']='issued'
    _update(policy['ledger'],operation)
