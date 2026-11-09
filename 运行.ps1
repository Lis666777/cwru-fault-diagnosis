# 默认执行一条示例诊断，避免误启动重新训练。
# infer为推理，audit为源码和数据检查，evaluate为冻结测试重放。
# prepare从随包原始MAT重建数据，随后核对参考数组。
# train在新的runs目录中重训，不替换正式推理权重。
# Python参数可指向另一台机器上准备好的兼容解释器。
# InputPath仅用于推理，要求12kHz的2048点驱动端波形。
# Name仅允许安全的目录名称，防止路径跳出交付目录。
# 所有工具均通过参数数组调用，不拼接可执行命令字符串。
# 输出位于reports，原始测试证据始终保留在qwen_v1。
# 本入口是可编辑源码，不是封装后的专有程序。
param(
    [ValidateSet('infer','audit','evaluate','prepare','train')][string]$Action='infer',
    [string]$Python='',
    [string]$InputPath='',
    [ValidatePattern('^[A-Za-z0-9_-]+$')][string]$Name='run1'
)
# 优先使用显式解释器，其次读取本机配置，最后查找PATH。
# local_runtime.json是便利配置，不属于模型依赖路径。
# 找不到本机配置中的解释器时不影响在另一台机器指定Python。
if (-not $Python) {
    $config=Join-Path $PSScriptRoot 'local_runtime.json'
    if (Test-Path -LiteralPath $config) {
        $candidate=(Get-Content -LiteralPath $config -Raw -Encoding UTF8 | ConvertFrom-Json).python
        if (Test-Path -LiteralPath $candidate) { $Python=$candidate }
    }
}
if (-not $Python) { $Python=(Get-Command python -ErrorAction Stop).Source }
# PowerShell 5.1调用中文脚本时，本文件保存为UTF-8 BOM。
# Python输出设为UTF-8，避免控制台错误信息出现乱码。
$env:PYTHONIOENCODING='utf-8'
switch ($Action) {
    'infer' {
        # 示例输入只含真实波形；其文件名不会进入模型。
        if (-not $InputPath) { $InputPath=Join-Path $PSScriptRoot 'examples/inner007_test_window.npy' }
        & $Python (Join-Path $PSScriptRoot 'predict.py') --input $InputPath --sample-rate 12000 --output (Join-Path $PSScriptRoot 'reports/diagnosis.json')
    }
    'audit' { & $Python (Join-Path $PSScriptRoot 'verify.py') }
    'evaluate' { & $Python (Join-Path $PSScriptRoot 'verify.py') --evaluate }
    'prepare' {
        # 新目录已存在时prepare报错，用户应更换Name保留历史结果。
        $target=Join-Path $PSScriptRoot "runs/rebuild_$Name"
        & $Python (Join-Path $PSScriptRoot 'prepare.py') --destination $target
        if ($LASTEXITCODE -ne 0) { throw '数据重建失败。' }
        & $Python (Join-Path $PSScriptRoot 'verify.py') --rebuilt $target
    }
    'train' {
        # 重新训练沿用冻结协议，不自动把新权重发布到推理入口。
        & $Python (Join-Path $PSScriptRoot 'llm.py') --run-dir (Join-Path $PSScriptRoot "runs/$Name")
    }
}
# 不吞掉子程序的异常或把失败显示成完成。
if ($LASTEXITCODE -ne 0) { throw "任务失败：$Action" }
