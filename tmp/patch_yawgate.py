# -*- coding: utf-8 -*-
import pathlib
p = pathlib.Path("/home/wfx/DR_competition/0810new/deeprobot_competition/src/S10_sdk_deploy/scripts/stair_vmc_noros.py")
src = p.read_text(encoding="utf-8-sig")
old = """                _stair_exec = abs(_d_first) < float(os.environ.get(
                    'S10_STAIR_EXEC_D', '0.5'))
            except Exception:
                _stair_exec = True"""
assert old in src
new = """                _stair_exec = abs(_d_first) < float(os.environ.get(
                    'S10_STAIR_EXEC_D', '0.5'))
                # v923: 对准门控——stair 执行区轮层冻结(纯前驱)+hip-yaw 弱，
                # 偏航进梯后纠不回来（原地图 yaw 1.14 vs 1.477 实测打转）。
                # 未对准(|yaw_err|>=门限)保持 CarVMC 完整轮控对准后再切。
                if _stair_exec:
                    _yaw_gate = float(os.environ.get(
                        'S10_STAIR_YAW_GATE', '0.12'))
                    _yaw_err_n = float(getattr(fol, '_last_err', 0.0))
                    if abs(_yaw_err_n) >= _yaw_gate:
                        _stair_exec = False
            except Exception:
                _stair_exec = True"""
src = src.replace(old, new, 1)
p.write_text(src, encoding="utf-8")
print("patched")