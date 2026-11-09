"""Prepare local runtime assets from a delivery bundle or pinned public sources."""
import argparse
import hashlib
import json
import shutil
import subprocess
import sys
from pathlib import Path

# 源码仓库不重复上传基础模型、公开数据或第三方依赖。
# 本工具提供离线接入交付包和从官方来源准备训练两种路径。
# 不访问或复制用户的实车原始日志、项目文档、账号凭据。
ROOT=Path(__file__).resolve().parent
REVISION='7ae557604adf67be50417f59c2c2f167def9a775'
MODEL_SHA='fdf756fa7fcbe7404d5c60e26bff1a0c8b8aa1f72ced49e7dd0210fe288fb7fe'

def sha(path):
    # 分块检查基础权重，避免将整个文件额外读入内存。
    h=hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda:stream.read(8*1024*1024),b''):
            h.update(block)
    return h.hexdigest()

def copy_new(source,target):
    # 已存在的文件或目录不覆盖，防止用户正在使用的权重丢失。
    # source由明确的交付目录和固定白名单拼接，不扫描整个工作区。
    if target.exists():
        raise FileExistsError(target)
    if source.is_dir():
        shutil.copytree(source,target,ignore=shutil.ignore_patterns('__pycache__','*.pyc','.cache'))
    else:
        shutil.copy2(source,target)

def main():
    # 离线模式用于复核已归档权重；公开模式用于从头重训。
    # 两种模式的用途不同，不能把新训练当作原权重的逐字复现。
    # 固定提交避免基础模型的最新版本变化影响实验。
    # SHA256不匹配时立即退出，不尝试忽略或修补校验错误。
    # 基础模型自带许可证，源码仓库许可证不替代它。
    # 本工具不创建账号、保存令牌或发布任何用户文件。
    # public_registry只包含公开下载链接、文件名及内容摘要。
    # raw、data、assets和运行输出均在gitignore中排除。
    # 网络请求只用于用户明确运行的公开准备模式。
    # 交付目录复制不修改源包，发生异常也不会删除源数据。
    # 数据构建子进程失败时直接传播非零退出状态。
    # 完成后用户仍需使用自己的兼容Python和NVIDIA运行环境。
    # 不指定离线包时，仅下载可公开获得的基础模型与CWRU原始数据。
    # 此路径没有本项目训练后的LoRA，需要用户随后运行训练入口。
    parser=argparse.ArgumentParser()
    parser.add_argument('--from-delivery',type=Path)
    args=parser.parse_args()
    if args.from_delivery:
        source=args.from_delivery.resolve()
        # 先核对核心权重，避免将错误目录当作已经验证的交付包。
        if sha(source/'assets/base_model/model.safetensors')!=MODEL_SHA:
            raise ValueError('Foundation model does not match the validated revision')
        names=['assets','data','raw','sources','qwen_v1','baseline_v1','examples',
               'protocol.json','registry.json','windows.json','data_audit.json']
        # 预先检查全部目标，避免到复制中途才遇到明显的同名冲突。
        for name in names:
            if (ROOT/name).exists():
                raise FileExistsError(ROOT/name)
            if not (source/name).exists():
                raise FileNotFoundError(source/name)
        for name in names:
            copy_new(source/name,ROOT/name)
    else:
        # huggingface_hub从已安装的固定版本依赖导入，不自动执行pip。
        from huggingface_hub import snapshot_download
        import requests
        # 固定提交和文件白名单；不下载或执行远程Python模型代码。
        snapshot_download('Qwen/Qwen2.5-0.5B-Instruct',revision=REVISION,
            local_dir=ROOT/'assets/base_model',token=False,
            allow_patterns=['config.json','generation_config.json','model.safetensors',
                            'tokenizer.json','tokenizer_config.json','merges.txt','vocab.json','LICENSE','README.md'])
        if sha(ROOT/'assets/base_model/model.safetensors')!=MODEL_SHA:
            raise ValueError('Foundation SHA256 mismatch')
        # 下载程序与训练程序分开，推理时仍保持离线运行。
        (ROOT/'raw').mkdir(exist_ok=True)
        (ROOT/'sources').mkdir(exist_ok=True)
        registry=json.loads((ROOT/'public_registry.json').read_text(encoding='utf-8'))
        # 原始文件哈希来自已经归档并复核的官方40文件注册表。
        # 不从任意第三方镜像下载，也不接受无法匹配的MAT文件。
        for row in registry:
            path=ROOT/'raw'/row['file']
            if not path.exists():
                response=requests.get(row['url'],timeout=90)
                response.raise_for_status()
                path.write_bytes(response.content)
            if sha(path)!=row['sha256']:
                raise ValueError('CWRU checksum mismatch: '+row['file'])
        # prepare直接读取官方页面，核对标签后保存页面作为本地证据。
        # 不将第三方网站完整HTML页面重新发布到源码仓库。
        # 数据仍写入新的构建目录，成功后才复制到训练入口的位置。
        build=ROOT/'runs/public_data_build'
        subprocess.run([sys.executable,str(ROOT/'prepare.py'),'--destination',str(build)],check=True)
        for name in ['data','protocol.json','registry.json','windows.json','data_audit.json']:
            copy_new(build/name,ROOT/name)
        # 训练生成的权重不从外部未知来源下载。
        # 固定实验成绩来自交付包，重新训练的成绩必须重新记录。
        print('Public training data ready. Run: python llm.py --run-dir qwen_v1')
    # 本机解释器路径仅写入已被gitignore排除的便利配置。
    # 这个路径不包含在公开源码包或上传文件白名单中。
    (ROOT/'reports').mkdir(exist_ok=True)
    (ROOT/'local_runtime.json').write_text(json.dumps({'python':sys.executable},indent=2),encoding='utf-8')
    print('ASSETS_READY')

if __name__=='__main__':
    main()
