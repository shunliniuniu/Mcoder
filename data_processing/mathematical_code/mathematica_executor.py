import subprocess
import os
import uuid
import re
import time
from concurrent.futures import ProcessPoolExecutor, TimeoutError
import logging
class MathematicaExecutor:
    def __init__(self, timeout=120, max_workers=3, temp_dir=None, wolframscript_path=None):
        self.timeout = timeout
        self.max_workers = max_workers
        self.temp_dir = temp_dir or os.environ.get('TEMP', r"D:\temp")
        self.wolframscript_path = wolframscript_path or r"D:\wolfram14.2\wolframscript.exe"
        self.execution_stats = {
            "total_executions": 0,
            "successful_executions": 0,
            "failed_executions": 0,
            "timeout_executions": 0,
            "total_execution_time": 0
        }
        self._setup_logging()
        self._validate_environment()
    def _setup_logging(self):
        log_dir = os.path.join(os.path.dirname(self.temp_dir), 'logs')
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)
        log_file = os.path.join(log_dir, 'mathematica_execution.log')
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.StreamHandler(),
                logging.FileHandler(log_file, encoding='utf-8')
            ]
        )
        self.logger = logging.getLogger(self.__class__.__name__)
        self.logger.info("Mathematica执行器初始化完成")

    def _validate_environment(self):
        if not os.path.exists(self.wolframscript_path):
            error_msg = f"WolframScript未找到: {self.wolframscript_path}"
            self.logger.error(error_msg)
            raise FileNotFoundError(error_msg)
        if not os.path.exists(self.temp_dir):
            try:
                os.makedirs(self.temp_dir)
                self.logger.info(f"创建临时目录: {self.temp_dir}")
            except Exception as e:
                error_msg = f"无法创建临时目录 {self.temp_dir}: {str(e)}"
                self.logger.error(error_msg)
                raise Exception(error_msg)

        self.logger.info(f"环境验证完成: WolframScript={self.wolframscript_path}, TempDir={self.temp_dir}")

    def clean_mathematica_code(self, code):
        if not code or not isinstance(code, str):
            self.logger.warning("输入代码为空或非字符串类型")
            return ""
        original_code = code
        original_length = len(code)
        try:
            code = code.strip()
            code = re.sub(r'```mathematica\s*', '', code, flags=re.IGNORECASE)
            code = re.sub(r'```\s*$', '', code, flags=re.MULTILINE)
            code = re.sub(r'\*\*([^*]+)\*\*', r'\1', code)
            code = re.sub(r'\*([^*]+)\*', r'\1', code)
            code = re.sub(r'<[^>]+>', '', code)
            lines = code.split('\n')
            cleaned_lines = []
            for line in lines:
                line = line.strip()
                if not line:
                    continue
                if re.match(r'^\(\*\s*[-=*]+\s*\*\)$', line):
                    continue
                cleaned_lines.append(line)
            code = '\n'.join(cleaned_lines)
            code = self._standardize_syntax(code)
            if not self._validate_mathematica_syntax(code):
                self.logger.warning(f"代码可能不包含有效的Mathematica语法: {code[:100]}...")
                code = self._attempt_code_repair(code)
            final_length = len(code)
            self.logger.debug(f"代码清理完成: {original_length} -> {final_length} 字符")
            return code
        except Exception as e:
            self.logger.error(f"代码清理失败: {str(e)}")
            return original_code

    def _standardize_syntax(self, code):
        lines = code.split('\n')
        standardized_lines = []
        for line in lines:
            line = line.strip()
            if not line:
                continue
            if (line and
                    not line.endswith(';') and
                    not line.endswith(']') and
                    not line.startswith('(*') and
                    '=' in line):
                line += ';'
            standardized_lines.append(line)
        return '\n'.join(standardized_lines)
    def _validate_mathematica_syntax(self, code):
        if not code:
            return False
        mathematica_patterns = [
            r'\w+\s*=',
            r'Print\s*\[',
            r'Solve\s*\[',
            r'Plot\s*\[',
            r'N\s*\[',
            r'Sqrt\s*\[',
            r'(Sin|Cos|Tan)\s*\[',
            r'\b(Pi|E|I)\b',
            r'\w+\s*\[.*?\]',
            r';\s*$',
            r'\{.*?\}',
            r'->'
        ]
        return any(re.search(pattern, code) for pattern in mathematica_patterns)

    def _attempt_code_repair(self, code):

        if not code:
            return ""
        if (code and
                not re.search(r'Print\s*\[', code) and
                not re.search(r'=', code) and
                len(code.strip()) < 200):
            if re.search(r'[\+\-\*/\^]', code) or re.search(r'\b(Sin|Cos|Sqrt|N)\b', code):
                repaired_code = f'Print[{code.strip()}];'
                self.logger.info(f"尝试修复代码: 添加Print包装")
                return repaired_code
        return code

    def execute_single(self, code):
        start_time = time.time()
        temp_file = None
        try:
            self.execution_stats["total_executions"] += 1
            cleaned_code = self.clean_mathematica_code(code)
            if not cleaned_code:
                error_msg = "清理后未发现有效的Mathematica代码"
                self.logger.warning(error_msg)
                self.execution_stats["failed_executions"] += 1
                return "", error_msg
            temp_file = os.path.join(self.temp_dir, f"mathematica_{uuid.uuid4().hex[:8]}.wls")
            with open(temp_file, "w", encoding='utf-8') as f:
                f.write(cleaned_code)
            self.logger.debug(f"执行Mathematica代码: {temp_file}")
            self.logger.debug(f"代码内容 ({len(cleaned_code)} 字符): {cleaned_code[:200]}...")
            result = subprocess.run(
                [self.wolframscript_path, "-file", temp_file],
                capture_output=True,
                text=True,
                timeout=self.timeout,
                encoding='utf-8'
            )
            execution_time = time.time() - start_time
            self.execution_stats["total_execution_time"] += execution_time
            if result.returncode == 0:
                output = result.stdout.strip()
                self.logger.info(f"执行成功: 输出长度={len(output)}, 耗时={execution_time:.2f}秒")
                self.execution_stats["successful_executions"] += 1
                return output, ""
            else:
                error_msg = result.stderr.strip() or f"执行失败，返回码: {result.returncode}"
                self.logger.error(f"执行失败: {error_msg}")
                self.execution_stats["failed_executions"] += 1
                return "", error_msg
        except subprocess.TimeoutExpired:
            execution_time = time.time() - start_time
            self.execution_stats["total_execution_time"] += execution_time
            self.execution_stats["timeout_executions"] += 1
            self.execution_stats["failed_executions"] += 1
            error_msg = f"执行超时 ({self.timeout}秒)"
            self.logger.error(error_msg)
            return "", error_msg
        except FileNotFoundError:
            error_msg = f"WolframScript未找到: {self.wolframscript_path}"
            self.logger.error(error_msg)
            self.execution_stats["failed_executions"] += 1
            return "", error_msg
        except Exception as e:
            execution_time = time.time() - start_time
            self.execution_stats["total_execution_time"] += execution_time
            self.execution_stats["failed_executions"] += 1
            error_msg = f"执行时发生意外错误: {str(e)}"
            self.logger.error(error_msg)
            return "", error_msg
        finally:
            if temp_file and os.path.exists(temp_file):
                try:
                    os.remove(temp_file)
                    self.logger.debug(f"删除临时文件: {temp_file}")
                except Exception as e:
                    self.logger.warning(f"删除临时文件失败 {temp_file}: {str(e)}")
    def batch_apply(self, batch_code):
        if not batch_code:
            self.logger.warning("批量执行: 代码列表为空")
            return []
        batch_size = len(batch_code)
        self.logger.info(f"开始批量执行: {batch_size} 个代码片段")
        start_time = time.time()
        results = []
        valid_codes = []
        code_indices = []
        for i, code in enumerate(batch_code):
            if code and isinstance(code, str) and code.strip():
                valid_codes.append(code)
                code_indices.append(i)
            else:
                self.logger.warning(f"跳过无效代码 (索引 {i})")
        if not valid_codes:
            self.logger.warning("批量执行: 没有有效代码")
            return [("", {"concise_exec_info": "无有效代码"}) for _ in batch_code]
        batch_results = [("", {"concise_exec_info": "未执行"}) for _ in batch_code]
        try:
            with ProcessPoolExecutor(max_workers=self.max_workers) as executor:
                future_to_index = {
                    executor.submit(self.execute_single, code): (i, code_indices[i])
                    for i, code in enumerate(valid_codes)
                }
                completed_count = 0
                for future in future_to_index:
                    valid_index, original_index = future_to_index[future]
                    try:
                        result = future.result(timeout=self.timeout + 10)
                        batch_results[original_index] = (result[0], {"concise_exec_info": result[1]})
                        completed_count += 1
                        if completed_count % 5 == 0:
                            self.logger.info(f"批量执行进度: {completed_count}/{len(valid_codes)}")
                    except TimeoutError:
                        error_msg = "批量执行超时"
                        self.logger.error(f"{error_msg} (索引 {original_index})")
                        batch_results[original_index] = ("", {"concise_exec_info": error_msg})
                    except Exception as e:
                        error_msg = f"批量执行错误: {str(e)[:100]}"
                        self.logger.error(f"{error_msg} (索引 {original_index})")
                        batch_results[original_index] = ("", {"concise_exec_info": error_msg})
        except Exception as e:
            self.logger.error(f"批量执行过程中发生严重错误: {str(e)}")
            for i in range(len(batch_code)):
                if batch_results[i][1]["concise_exec_info"] == "未执行":
                    batch_results[i] = ("", {"concise_exec_info": f"批量执行失败: {str(e)[:50]}"})
        batch_time = time.time() - start_time
        successful_count = sum(1 for result in batch_results if result[0] and not result[1]["concise_exec_info"])
        failed_count = batch_size - successful_count
        self.logger.info(f"批量执行完成:")
        self.logger.info(f"总任务数: {batch_size}")
        self.logger.info(f"成功执行: {successful_count}")
        self.logger.info(f"执行失败: {failed_count}")
        self.logger.info(f"成功率: {successful_count / batch_size * 100:.1f}%")
        self.logger.info(f"总耗时: {batch_time:.2f}秒")
        return batch_results

    def apply(self, code):
        result = self.execute_single(code)
        return (result[0], {"concise_exec_info": result[1]})
    def get_stats(self):
        stats = {
            "配置信息": {
                "timeout": self.timeout,
                "max_workers": self.max_workers,
                "temp_dir": self.temp_dir,
                "wolframscript_path": self.wolframscript_path
            },
            "执行统计": self.execution_stats.copy()
        }
        if self.execution_stats["total_executions"] > 0:
            avg_time = self.execution_stats["total_execution_time"] / self.execution_stats["total_executions"]
            stats["执行统计"]["average_execution_time"] = round(avg_time, 2)
            success_rate = self.execution_stats["successful_executions"] / self.execution_stats[
                "total_executions"] * 100
            stats["执行统计"]["success_rate"] = round(success_rate, 1)
        return stats
    def reset_stats(self):
        self.execution_stats = {
            "total_executions": 0,
            "successful_executions": 0,
            "failed_executions": 0,
            "timeout_executions": 0,
            "total_execution_time": 0
        }
        self.logger.info("执行统计已重置")

    def __del__(self):
        if hasattr(self, 'logger'):
            self.logger.info("Mathematica执行器正在销毁")
