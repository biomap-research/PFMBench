from baidubce.bce_client_configuration import BceClientConfiguration
from baidubce.auth.bce_credentials import BceCredentials
from baidubce.services.bos.bos_client import BosClient
import hashlib
import glob

def seq_encoder(sequence, method='md5'):
    hasher = eval(f'hashlib.{method}')
    return hasher(sequence.encode(encoding='utf-8')).hexdigest()


config = BceClientConfiguration(
    credentials = BceCredentials(
        '35420270cb5c46118d6729b692669e2b',
        '35474e577b514954b72a128a53304cab'
    ),
    endpoint = 'https://bj.bcebos.com'
)

bos_client = BosClient(config)

# response = bos_client.list_buckets()
# for bucket in response.buckets:
# 	 print(bucket.name)


if __name__ == "__main__":
    import pandas as pd
    all_csvs = list(
        glob.iglob(
            "/nfs_beijing/kubeflow-user/zhangyang_2024/workspace/protein_benchmark/datasets/*/*.csv",
            recursive=True
        )
    )
    all_csvs.sort()

    seqs = []
    pdb_paths = []

    for _csv in all_csvs:
       df = pd.read_csv(_csv)
       if "pdb_path" not in df or "aa_seq" not in df:
            print(_csv)
            continue
       _seqs = df["aa_seq"].tolist()
       _pdb_paths = df["pdb_path"].tolist()
       seqs.extend(_seqs)
       pdb_paths.extend(_pdb_paths)
    
    print(len(pdb_paths))
    print(len(seqs))
