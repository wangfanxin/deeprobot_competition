#!/bin/bash
# NmpcWbc 闂傚倸鍊搁崐鐑芥倿閿曗偓椤啴骞愭惔锝庢锤闂佺粯鍔曢幖顐ょ玻濡ゅ懎绠归弶鍫濆⒔缁嬪鏌￠崱妯肩煁闁靛洤瀚伴獮鍥礈娴ｇ儤娈搁梻?wp6->7 (v1083: 闂傚倸鍊搁崐宄懊归崶顒夋晪闁哄稁鍘奸崒銊ф喐閻楀牆绗掗柛銊ュ€婚幉鎼佹偋閸繂鎯為梺鎼炲労閸撴瑩鎷戦悢鍏肩厪濠㈣泛鐗嗛崝銈嗐亜閵忕姵鍣虹紒缁樼箞閹粙妫冨ù韬插灲閺屾盯鎮ゆ担鍝ヤ桓閻庢鍠栭崯鍧椻€﹂妸鈺侀唶闁绘柨鎼獮?+ 闂傚倸鍊搁崐椋庣矆娓氣偓楠炴牠顢曢敃鈧悿顕€鏌ｅΔ鈧悧濠囧矗韫囨稒鈷掗柛顐ゅ枍缁堕亶鏌ｉ幒鏇炐撻柕鍥у瀵粙濡歌閳ь剚甯楅妵鍕煛閸滀焦顥栫紓浣介哺鐢€崇暦閹烘埈鐓ラ柟閭﹀灠閹潐es闂傚倸鍊搁崐鎼佸磹閹间礁纾归柛婵勫劗閸嬫挸顫濋悡搴＄睄閻庤娲樼换鍫濐嚕閹绢喗鍋愰柛蹇撴憸閻?+ SWING闂傚倸鍊搁崐椋庣矆娓氣偓楠炲鏁撻悩鑼唶闂佸憡绺块崕鎶芥儗閹剧粯鐓欑紓浣靛灩閺嬫稓绱掗悩宕囧缂佺粯鐩獮妯尖偓鐢殿焾鎼村y + 闂傚倸鍊峰ù鍥х暦閻㈢纾婚柣鎰暩閻瑩鐓崶銊р槈缂佲偓婢舵劖鐓欓悗娑欘焽缁犮儵鏌ｉ幒鏇燁棄闂囧鏌ｉ幘铏崳闁圭晫濞€閺屾稒鎯旈埄鍐╁闯缂備浇椴哥敮鐐哄焵椤掑﹦绉甸柛瀣浮瀹曟洟濡烽埡鍌滃幈闂侀潧鐗嗗ú銈呪枔閻戠うER)
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
export S10_NMPC_WF=1e-3 S10_NMPC_WFR=0.005 S10_NMPC_WM=0.3 S10_NMPC_PITCH_FF_SW=1.5
export S10_NMPC_WA=1.0 S10_NMPC_Z_OFF=0.25 S10_NMPC_KP_VX=10.0
export S10_NMPC_KP_YAW=6.0 S10_NMPC_YAW_ERR_K=1.0 S10_NMPC_WHEEL_K=12.0 S10_NMPC_YAW_DIFF=1.0
export S10_NMPC_DEBUG=1 S10_NMPC_HOVER_LEN=0.10 S10_NMPC_HOVER_TMAX=0.5
export S10_NMPC_AZ_MIN=-4.0 S10_NMPC_AZ_MAX=6.0 S10_NMPC_AL_LIM=30.0
export S10_STAIR_HDG_K=1.0 S10_STAIR_HDG_D=3.0 S10_STAIR_HDG_KI=0.15
export S10_STAIR_HDG_OM=0.5 S10_STAIR_HDG_LAT=0.35 S10_STAIR_OM_SCALE=1.0
export VMC_TRAJ=/home/wfx/DR_competition/0810new/deeprobot_competition/tmp/traj_nmpc_real37.npy
exec bash tmp/run_stw_smoke.sh