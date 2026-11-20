"""Export frozen pre-vote landmark responses for downstream ICL."""
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
from .data import read_table
from .predict import load_artifact


def landmark_responses(artifact, x, chunk_size=256):
    if not artifact.get('kernels'):
        raise ValueError('A native classification kernel export is required.')
    if chunk_size < 1:
        raise ValueError('chunk_size must be positive')
    parts=[]
    for start in range(0,len(x),chunk_size):
        raw=x[start:start+chunk_size]
        z=np.nan_to_num((raw-artifact['mean'])/artifact['std'])
        missing=~np.isfinite(raw)
        classes=[]
        for kernel in artifact['kernels']:
            distance=np.where(missing[:,None,:],np.asarray(kernel['missing'])[None,:,:],
                np.asarray(kernel['B'])[None,:,:]*(z[:,None,:]-kernel['center'])**2).sum(2)
            classes.append(np.exp(-distance))
        parts.append(np.concatenate(classes,axis=1))
    width=sum(len(k['coef']) for k in artifact['kernels'])
    return np.concatenate(parts,axis=0) if parts else np.empty((0,width))


def feature_frame(directory, frame, combine=False):
    encoder,artifact,_=load_artifact(directory)
    x=encoder.transform(frame)
    names=[f'landmark_c{c}_{l}' for c,k in enumerate(artifact['kernels']) for l in range(len(k['coef']))]
    result=pd.DataFrame(landmark_responses(artifact,x),columns=names)
    if combine:
        original=pd.DataFrame(x,columns=['original:'+name for name in encoder.names])
        result=pd.concat([original,result],axis=1)
    return result


def main():
    parser=argparse.ArgumentParser(description='Export frozen landmark responses, optionally concatenated with original encoded features.')
    parser.add_argument('run')
    parser.add_argument('csv')
    parser.add_argument('--output',required=True)
    parser.add_argument('--combine',action='store_true')
    args=parser.parse_args()
    output=Path(args.output)
    if output.exists():parser.error('Choose a new output file.')
    result=feature_frame(args.run,read_table(args.csv),args.combine)
    output.parent.mkdir(parents=True,exist_ok=True)
    with output.open('x',encoding='utf-8',newline='') as stream:result.to_csv(stream,index=False)
    print(f'Exported {len(result)} rows and {len(result.columns)} features.')


if __name__=='__main__':main()
