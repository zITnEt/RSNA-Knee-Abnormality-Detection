"""After DICOM preprocessing, queue bounded lossless publication bridges."""
import argparse,json,os,subprocess,time
from pathlib import Path
from kaggle_monitor_runtime import awake_on_ac
from task_state_io import atomic_json_write
ROOT=Path(__file__).resolve().parents[1];REPORT=ROOT/'results/min_overlap_384'
STATE=REPORT/'bridge_queue_state.json'


def command(args):
    try:
        r=subprocess.run(['kaggle',*args],capture_output=True,text=True,encoding='utf-8',errors='replace',timeout=120,cwd=ROOT)
        return r.returncode,r.stdout+r.stderr
    except subprocess.TimeoutExpired:return 124,'Temporary timeout'


def persist(state):
    state['updated_utc']=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime())
    atomic_json_write(STATE,state)


def main(hours=24):
    REPORT.mkdir(parents=True,exist_ok=True);(REPORT/'bridge_monitor_pid.txt').write_text(str(os.getpid()))
    state=json.loads(STATE.read_text()) if STATE.exists() else dict(jobs=[dict(shard=i,state='pending',
        ref=f'zafarnematov/rsna-series-safe-384-preprocessing-{i:02d}') for i in range(32)],complete=False)
    end=time.time()+hours*3600
    while time.time()<end:
        try:
            processing=REPORT/'queue_state.json'
            processing_state=json.loads(processing.read_text()) if processing.exists() else {}
            source_ready={j['shard'] for j in processing_state.get('jobs',[]) if j['state']=='complete'}
            source_active=sum(j['state'] in ('queued','running','receipt_wait') for j in processing_state.get('jobs',[]))
            if not source_ready:
                state['waiting_for_preprocessing']=True;persist(state);time.sleep(60);continue
            state['waiting_for_preprocessing']=not processing_state.get('complete',False)
            for job in state['jobs']:
                if job['state'] not in ('queued','running','receipt_wait'):continue
                code,status=command(['kernels','status',job['ref']]);job['last_status']=status.strip()
                if code:continue
                if 'ERROR' in status:raise RuntimeError('Publication bridge failed '+str(job['shard']))
                if 'COMPLETE' in status:
                    folder=REPORT/f"bridge_{job['shard']:02d}"
                    code,message=command(['kernels','output',job['ref'],'-p',str(folder),'--file-pattern',
                        r'manifest\.json$|recipe\.json$|publication_validation\.json$','--page-size','200','-q','-o'])
                    reports=list(folder.rglob('publication_validation.json'));job['state']='receipt_wait'
                    if code or len(reports)!=1:continue
                    report=json.loads(reports[0].read_text());assert report['complete'] and report['all_decompressed_array_bytes_identical']
                    job.update(state='complete',studies=report['studies'],bytes=report['bytes'])
                    print('PUBLICATION_BRIDGE_VALIDATED '+json.dumps(job),flush=True)
                else:job['state']='running' if 'RUNNING' in status else 'queued'
            active=sum(j['state'] in ('queued','running','receipt_wait') for j in state['jobs'])
            for job in state['jobs']:
                if active>=max(0,5-source_active):break
                if job['state']!='pending':continue
                if job['shard'] not in source_ready:continue
                code,out=command(['kernels','push','-p',str(ROOT/f"kaggle/min_overlap_384_publication/shard_{job['shard']:02d}"),'-t','10800'])
                job['last_push']=out.strip()
                if not code and 'successfully pushed' in out:
                    job['state']='queued';active+=1;persist(state);print('BRIDGE_LAUNCHED '+job['ref'],flush=True)
                elif code==124 or any(w in out.lower() for w in ('session count','sessioncount','maximum number','cpu sessions','capacity','connection','nameresolution')):break
                else:raise RuntimeError(out[-1500:])
            state['completed_shards']=sum(j['state']=='complete' for j in state['jobs'])
            state['complete']=state['completed_shards']==32;persist(state)
            if state['complete']:
                state['bytes']=sum(j['bytes'] for j in state['jobs']);state['studies']=sum(j['studies'] for j in state['jobs'])
                assert state['studies']==4407;persist(state);print('ALL_PUBLICATION_BRIDGES_COMPLETE '+str(state['bytes']),flush=True);return
        except Exception as exc:
            state['needs_attention']=True;state['error']=str(exc);persist(state);print('BRIDGE_MONITOR_STOP '+str(exc),flush=True);return
        time.sleep(60)
    state['monitor_expired']=True;persist(state)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--hours',type=float,default=24)
    with awake_on_ac():main(**vars(parser.parse_args()))
