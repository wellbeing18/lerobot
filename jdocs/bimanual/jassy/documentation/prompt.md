1. I notice that the scanner prints "  [OK] Left follower: /dev/ttyACM3
  [OK] Right follower: /dev/ttyACM2
  [OK] Left leader: /dev/ttyACM0
  [OK] Right leader: /dev/ttyACM1
  [OK] Head camera: /dev/video4
  [OK] Left wrist: /dev/video8
  [OK] Right wrist: /dev/video6" but is this just checking whether the dev ports 0-3 exist? Or also checking which arm is connected to which port? Because the ports 0-3 are always connected but the issue is, for example, overnight port 0 might switch from left leader arm to right leader arm. 2. 



  1. both arms are currently in a folded home position
  2. I can 100% guarantee that ACM3 is left follower and ACM2 is right follower. You also suggest that the caliibration files may have been created with ports assigned differently