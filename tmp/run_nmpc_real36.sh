#!/bin/bash
# NmpcWbc 闂傚倸鍊烽悞锕€顪冮幐搴ｎ洸闁绘劕鎼粣妤呭箹鏉堝墽绋婚柡鍡樼矋閵囧嫰骞囬崜浣烘殸闂?wp6->7 (v1083: 闂傚倸鍊峰ù鍥敋閺嶎厼鍌ㄧ憸鐗堝笒閸ㄥ倻鎲搁悧鍫濆惞闁搞儺鍓欓拑鐔兼煏婢跺牆鍔ゆい銏犳噺缁绘繈鎮介棃娴躲垽鏌涢悤浣哥仸鐎殿喖鍟块…銊╁醇閻斿搫骞?+ 闂傚倸鍊风粈渚€骞栭锕€鐤柣妤€鐗婇崣蹇涙⒒閸喍绶遍柣鎺曞Г閵囧嫰寮介妸褋鈧帗銇勯埡鍜佹缂佽鲸甯″畷鎺戭煥閹邦垰鎯坉es闂傚倸鍊搁崐鎼佸磹閸濄儮鍋撳鐓庡籍鐎规洘绻堝鎾偐閸忓摜鐟?+ SWING闂傚倸鍊风粈渚€骞夐敓鐘茬闁告縿鍎抽惌鎾绘煙缂併垹鏋涚紒鐘崇墬缁绘盯骞樼€电搴妉y + 闂傚倷娴囧畷鐢稿磻閻愮數鐭欓煫鍥ㄧ☉缁€澶愭煙鐎涙绠ラ柣鎺曟闇夐柣鎾虫捣閹界娀鏌涙惔鈩冩崳缂佽鲸甯為埀顒婄秵閸嬫帡宕曢妷鈺傜厱闁靛牆娲ゅ▓鐑礦ER)
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
export S10_NMPC_KP_YAW=6.0 S10_NMPC_YAW_ERR_K=1.0 S10_NMPC_WHEEL_K=12.0 S10_NMPC_YAW_DIFF=1.0
export S10_NMPC_DEBUG=1 S10_NMPC_HOVER_LEN=0.10 S10_NMPC_HOVER_TMAX=0.5
export S10_NMPC_AZ_MIN=-4.0 S10_NMPC_AZ_MAX=6.0 S10_NMPC_AL_LIM=30.0
export S10_STAIR_HDG_K=1.0 S10_STAIR_HDG_D=3.0 S10_STAIR_HDG_KI=0.15
export S10_STAIR_HDG_OM=0.5 S10_STAIR_HDG_LAT=0.35 S10_STAIR_OM_SCALE=1.0
export VMC_TRAJ=/home/wfx/DR_competition/0810new/deeprobot_competition/tmp/traj_nmpc_real36.npy
exec bash tmp/run_stw_smoke.sh