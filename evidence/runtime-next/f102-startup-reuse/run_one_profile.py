from pathlib import Path
import hashlib,json,os,resource,signal,subprocess,sys,time
base=Path('/xcf26/zpzeng/practice_arena/astrolabe_runtime_20261102_implementation/cloud-20261104')
index=int(sys.argv[1])
assert 0 <= index < 4
output=base/'control/a-startup-profile-01'
if index == 0: output.mkdir()
else: assert output.is_dir()
variants=['before','candidate','candidate','before']
heads={'before':'47c6f6dd18fe822a6d28ea6970c756d5dfae7b95','candidate':'f9ce87b9f5cf37710f0a52ebe503595e5d7fdd7f'}
trees={'before':'f63f4e680d946cc30fd3981fe31fbf5aa4be4d12','candidate':'00cde2af9ed9c2a4778cf1e4e82e5a9e18183aba'}
repos={'before':base/'startup-baseline-47c6f6d','candidate':base/'startup-candidate-f9ce87b'}
profile=base/'control/profile_cold_startup.py'
assert hashlib.sha256(profile.read_bytes()).hexdigest()=='9d1580d22c779677527154596d5295b2df89a83acd0da516beb425a8f359be9b'
prior=[json.loads((output/('%02d-result.json'%i)).read_text()) for i in range(index)]
assert all(r['exit_code']==0 and r['status']=='DIAGNOSTIC_COMPLETE' for r in prior)
used_cpu=sum(r['total_cpu_seconds'] for r in prior)
assert used_cpu <= 68, 'insufficient remaining CPU budget for another bounded invocation'
prefix='%02d-%s'%(index,variants[index])
assert not (output/(prefix+'.started.json')).exists()
repo=repos[variants[index]]
def cpu_total():
 u=resource.getrusage(resource.RUSAGE_SELF);c=resource.getrusage(resource.RUSAGE_CHILDREN)
 return u.ru_utime+u.ru_stime+c.ru_utime+c.ru_stime
cpu_start=cpu_total()
observed_head=subprocess.check_output(['git','rev-parse','HEAD'],cwd=str(repo),universal_newlines=True).strip()
observed_tree=subprocess.check_output(['git','rev-parse','HEAD^{tree}'],cwd=str(repo),universal_newlines=True).strip()
assert observed_head==heads[variants[index]] and observed_tree==trees[variants[index]]
assert subprocess.call(['git','diff','--quiet','HEAD'],cwd=str(repo))==0
tmp=output/'tmp'
tmp.mkdir(exist_ok=True)
(tmp/'apptainer').mkdir(exist_ok=True)
env=dict(os.environ)
env.update(TMPDIR=str(tmp),APPTAINER_TMPDIR=str(tmp/'apptainer'),APPTAINERENV_TMPDIR=str(tmp),PYTHONDONTWRITEBYTECODE='1')
command=[str(base/'control/test-container-cpu'),str(repo),'-X','importtime',str(profile),'--repo',str(repo),'--variant',variants[index]]
record={'index':index,'variant':variants[index],'head':observed_head,'tree':observed_tree,'command':command,'start_unix':time.time(),'prior_cpu_seconds':used_cpu,'cpu_budget_seconds':90,'scope':'Cold CPU imports/source scanning only; zero scientific evaluations, no native init or GPU','profile_sha256':hashlib.sha256(profile.read_bytes()).hexdigest()}
(output/(prefix+'.started.json')).write_text(json.dumps(record,indent=2)+'\n')
if index==0:
 (output/'plan.json').write_text(json.dumps({'order':variants,'heads':heads,'trees':trees,'max_invocations':4,'hard_cpu_seconds_per_python':20,'total_cpu_budget_seconds':90,'max_wall_seconds_per_invocation':35,'kill_grace_seconds':2},indent=2)+'\n')
start=time.monotonic()
stop_reason=None
with (output/(prefix+'.stdout.log')).open('wb') as stdout,(output/(prefix+'.stderr.log')).open('wb') as stderr:
 proc=subprocess.Popen(command,cwd=str(repo),env=env,stdout=stdout,stderr=stderr,start_new_session=True)
 try:
  exit_code=proc.wait(timeout=35)
 except subprocess.TimeoutExpired:
  stop_reason='profile_wall_budget'
  if proc.poll() is None:
   try: os.killpg(proc.pid,signal.SIGTERM)
   except ProcessLookupError: pass
  try: exit_code=proc.wait(timeout=2)
  except subprocess.TimeoutExpired:
   if proc.poll() is None:
    try: os.killpg(proc.pid,signal.SIGKILL)
    except ProcessLookupError: pass
   exit_code=proc.wait(timeout=2)
wall=time.monotonic()-start
total_cpu=cpu_total()-cpu_start
text=(output/(prefix+'.stdout.log')).read_text(errors='replace')
phases=[]
for line in text.splitlines():
 try: row=json.loads(line)
 except ValueError: continue
 if isinstance(row,dict): phases.append(row)
complete=any(row.get('status')=='DIAGNOSTIC_COMPLETE' for row in phases)
record.update(exit_code=exit_code,stop_reason=stop_reason,wall_seconds=wall,total_cpu_seconds=total_cpu,cumulative_cpu_seconds=used_cpu+total_cpu,end_unix=time.time(),status='DIAGNOSTIC_COMPLETE' if exit_code==0 and complete and used_cpu+total_cpu<=90 else 'FAILED',phases=phases)
record['files']={p.name:{'bytes':p.stat().st_size,'sha256':hashlib.sha256(p.read_bytes()).hexdigest()} for p in [output/(prefix+'.stdout.log'),output/(prefix+'.stderr.log')]}
(output/('%02d-result.json'%index)).write_text(json.dumps(record,indent=2)+'\n')
brief={k:record[k] for k in ('index','variant','exit_code','status','wall_seconds','total_cpu_seconds','cumulative_cpu_seconds','stop_reason')}
brief['phases']=[{k:v for k,v in row.items() if k in ('phase','wall_seconds','cpu_seconds','maxrss_kib','pyrosetta_loaded','spatial_loaded','registry_count','status')} for row in phases]
brief['output']=str(output)
print(json.dumps(brief))
sys.exit(0 if record['status']=='DIAGNOSTIC_COMPLETE' else 1)
