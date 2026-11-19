"""Use the frozen ImageNet membership and labels with original ZIP image bytes."""
import csv,hashlib,io,json,zipfile,zlib
from pathlib import Path

def frozen_zip_index(archive, marker, expected, canonical, limit):
    original=json.loads(Path(marker).read_text())
    classes=original['classes'];rows=original['rows']
    if len(rows)!=expected['images'] or len(classes)!=expected['classes']:
        raise ValueError('Frozen ImageNet membership changed')
    with zipfile.ZipFile(archive) as source:
        raw=source.read('LOC_val_solution.csv')
        if hashlib.sha256(raw).hexdigest()!=original['annotation_sha256']:
            raise ValueError('ImageNet annotation checksum mismatch')
        labels={};class_ids={name:i for i,name in enumerate(classes)}
        for entry in csv.DictReader(io.StringIO(raw.decode())):
            synsets=set(entry['PredictionString'].split()[::5])
            if len(synsets)!=1 or entry['ImageId'] in labels:
                raise ValueError('Duplicate or ambiguous ImageNet annotation')
            labels[entry['ImageId']]=class_ids[synsets.pop()]
        if len({r['id'] for r in rows})!=len(rows) or set(labels)!={r['id'] for r in rows}:
            raise ValueError('ImageNet sample ID membership changed')
        relocated=[]
        for i,row in enumerate(rows):
            member=f"ILSVRC/Data/CLS-LOC/val/{row['id']}.JPEG"
            if labels[row['id']]!=row['label'] or source.getinfo(member).CRC!=row['crc32']:
                raise ValueError('ImageNet label or image CRC mismatch: '+member)
            if not limit or i<limit:
                if zlib.crc32(source.read(member))!=row['crc32']:
                    raise ValueError('Selected ImageNet image bytes changed: '+member)
            relocated.append({k:v for k,v in row.items() if k!='path'} | {'archive':str(archive),'member':member})
    result={**original,'rows':relocated,'sources':[{'path':str(archive),'annotation_sha256':original['annotation_sha256']}],
            'input_format':'zip','frozen_index_sha256':hashlib.sha256(Path(marker).read_bytes()).hexdigest(),
            'image_bytes_validated':len(rows) if not limit else min(limit,len(rows))}
    result.pop('fingerprint',None);result['fingerprint']=canonical(result)
    return result
