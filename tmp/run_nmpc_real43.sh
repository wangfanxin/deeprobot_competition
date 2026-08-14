#!/bin/bash
# NmpcWbc 闂傚倸鍊搁崐鎼佸磹閻戣姤鍊块柨鏇楀亾妞ゎ亜鍟撮獮鎰償閿濆孩閿ら梻浣虹帛閸旀洟骞栭銈囩幓婵°倕鎳庣粻褰掑级閸繂鈷旂紒瀣煼閺岋繝宕卞Ο鑲╃厑闂侀潧娲ょ€氫即鐛崶顒€绀堝ù锝囧劋濞堟悂姊?wp6->7 (v1083: 闂傚倸鍊搁崐鎼佸磹瀹勬噴褰掑炊椤掑鏅梺鍝勭▉閸樺ジ宕掗妸褎鍠愰柣妤€鐗嗙粭鎺楁煕閵娿儱鈧骞夐幖浣瑰亱闁割偅绻傞幆鐐烘⒑閹肩偛鍔撮柛鎾寸懇閹锋垿鎮㈤崗鑲╁帾婵犮垼娉涢悧鍡涘礉閵堝棎浜滈柕蹇曞У閸ｈ櫣绱掔紒妯肩疄闁诡喕绮欏Λ鍐归煬鎻掔伈闁哄本鐩幃銈嗘媴閸濄儰妗撻柣搴㈩問閸犳牠宕崸妞烩偓锕傚Ω閳轰線鍞堕梺缁樻煥閹碱偊鐛?+ 闂傚倸鍊搁崐鎼佸磹妞嬪海鐭嗗〒姘ｅ亾妤犵偞鐗犻、鏇㈡晝閳ь剟鎮块鈧弻锝呂旈埀顒勬偋婵犲洤鐭楅煫鍥ㄧ⊕閳锋帡鏌涢銈呮瀺缂佸爼浜堕弻锝夊箳閺囩倫鎾绘煏閸パ冾伃鐎殿喕绮欐俊姝岊槹闁逞屽墯鐢濡甸崟顖氱厸闁告粈鐒﹂ˉ鏍磽娴ｄ粙鍝洪悽顖椻偓宕囨殾闁圭儤鍩堥悡銉╂煙闁箑鐏犻柟顖氭綈es闂傚倸鍊搁崐鎼佸磹閹间礁纾归柟闂寸绾惧綊鏌涘┑鍕姉闁稿鎸搁～婵嬫偂鎼达紕鐫勯柣搴ゎ潐濞叉鎹㈤崼婵愬殨闁圭虎鍠楅崑鎰版煕韫囨挻鎲搁柣?+ SWING闂傚倸鍊搁崐鎼佸磹妞嬪海鐭嗗〒姘ｅ亾妤犵偛顦甸弫鎾绘偐閼碱剦鍞堕梻浣告啞缁哄潡宕曢幎鑺ュ剹闁瑰墽绮悡娆戠磽娴ｉ潧鐏╅柡瀣〒缁辨帡鎮╁畷鍥ь潷缂備胶绮惄顖炵嵁濡皷鍋撻悽娈跨劸閹兼潙顩畒 + 闂傚倸鍊搁崐宄懊归崶褏鏆﹂柣銏㈩焾绾惧鏌ｉ幇顔芥毄闁活厽鐟╅悡顐﹀炊閵娧€妲堢紓浣插亾濠㈣埖鍔栭悡娆撴倵濞戞瑯鐒界紒鐘劦閺岋綁骞掗弴鐕佹闂傚洤顦甸弻锝夊箻閾忣偅宕抽梺鍦櫕婵炩偓闁哄本绋掗幆鏃堝焺閸愨晛闂紓鍌欐祰妞村摜鏁悙鍝勭劦妞ゆ帒锕︾粔鐢告煕鐎ｎ偄娴€规洘娲熸俊鐑藉煛閸屾粌骞堥梻渚€娼ч悧鍡椕洪妶鍛灁闁绘垹銇咵R)
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
export VMC_TRAJ=/home/wfx/DR_competition/0810new/deeprobot_competition/tmp/traj_nmpc_real43.npy
exec bash tmp/run_stw_smoke.sh