#!/bin/bash
# NmpcWbc 闂傚倷鐒﹀鎸庣閻愬搫绐楅幖杈剧稻閺嗘粍銇勯幇鍓佺暠闁?wp6->7 (v1083: 闂傚倷娴囬鏍储瑜版帒鍨傜憸鐗堝吹閸ヮ剙钃熼柕澶堝劤椤㈠懏绻濋悽闈浶㈤柛鐕佸灦瀵啿顭ㄩ崼鐔哄幍?+ 闂傚倷绀侀幖顐﹀疮閻楀牊鍙忛梻鍫熶緱閻掕姤銇勯弽銊с€掓い鈺咁棑缁辨帡宕掑鎰惈des闂傚倸鍊搁崐鍝モ偓姘煎弮瀹曟繈寮撮悩鍏哥瑝?+ SWING闂傚倷绀侀幉锟犲箰閸濄儳鐭撻柟缁㈠枛缁犳牗绻涢幘瀵稿床ly + 闂備浇宕甸崑鐐电矙韫囨稑绀夐柟瀛樼箥閻掕棄霉閻撳海鎽犻柛搴℃捣缁辨帞鈧綆鍋掗崕銉╂煕閵堝洤娈烵VER)
cd /home/wfx/DR_competition/0810new/deeprobot_competition
export S10_VMC_MODE=nmpcwbc S10_STAIR_BENCH=0 S10_START_WP=6 S10_START_BACK=2.0
export S10_AUTO_MAX_WP=8 S10_TEST_MAX_SIM=60 S10_NAV_HZ=20
export S10_STUCK_TIMEOUT=25 S10_VMC_TERRAIN=lidar
export S10_STAIR_POSMODE=1 S10_STAIR_SWING_D=0.35 S10_STAIR_ENTER_DIST=2.0
export S10_STAIR_VX_RAMP=10.0 S10_STAIR_WIN_VX=1.0 S10_STAIR_EXEC_D=1.0
export S10_STAIR_MPPI_OFF_D=0.5 S10_STAIR_CREST_VX=1.0
export S10_AUTO_LOOKAHEAD_STAIR=3.5 S10_AUTO_CTE_GAIN_STAIR=1.0 S10_AUTO_YAW_GAIN_STAIR=1.5
export S10_YAW_DAMP=2.0 S10_STAIR_YAW_GATE=1.0 S10_VMC_TERRAIN_KIN=1
export S10_CAR_YAW_K_SM=10 S10_VMC_YAW_K_WHEEL=20 S10_CAR_YAW_SLEW=12
export S10_FP_STAND_DROP=0.22 S10_WHEEL_PRESS=0.05
export S10_NMPC_HZ=20 S10_NMPC_SWING_D=0.35 S10_NMPC_MU=0.8
export S10_NMPC_WF=1e-3 S10_NMPC_WM=0.3 S10_NMPC_PITCH_FF_SW=1.5
export S10_NMPC_WA=1.0 S10_NMPC_Z_OFF=0.25 S10_NMPC_KP_VX=10.0
export S10_NMPC_KP_YAW=6.0 S10_NMPC_YAW_ERR_K=1.5 S10_NMPC_YAW_DIFF=2.0
export S10_NMPC_DEBUG=1 S10_NMPC_HOVER_LEN=0.10 S10_NMPC_HOVER_TMAX=0.5
export S10_NMPC_AZ_MIN=-4.0 S10_NMPC_AZ_MAX=12.0 S10_NMPC_AL_LIM=30.0
export S10_STAIR_HDG_K=1.0 S10_STAIR_HDG_D=3.0 S10_STAIR_HDG_KI=0.15
export S10_STAIR_HDG_OM=0.5 S10_STAIR_HDG_LAT=0.35 S10_STAIR_OM_SCALE=1.0
export VMC_TRAJ=/home/wfx/DR_competition/0810new/deeprobot_competition/tmp/traj_nmpc_real33.npy
exec bash tmp/run_stw_smoke.sh