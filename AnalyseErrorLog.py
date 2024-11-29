import re
import pandas as pd
from datetime import datetime



# 2024-05-30_18-11-49.412	ID: Tamer	Time: 18:11:49.398 30/05/2024	Error: TRUE	Code: 5801	 Description: newtrapctrlhost: Tamer: FPGA returned Retry More info: 5TC1
test_lines = ["2024-05-30_18-11-49.412	ID: Tamer	Time: 18:11:49.398 30/05/2024	Error: TRUE	Code: 5801	 Description: newtrapctrlhost: Tamer: FPGA returned Retry More info: 5TC1",
                  "2024-05-30_09-22-09.444	ID: 5TC1 Kasli Listener	Time: 09:22:09.444 30/05/2024	Error: TRUE	Code: 66	 Description: SUBSTITUTED: TCP Read in TALOS_PPL.lvlibp:TCP Listener.lvlib:TCP Listener.lvclass:Read Message Length.vi:6510001->TALOS_PPL.lvlibp:TCP Listener.lvlib:TCP Listener.lvclass:Consumer.vi:4400004->TALOS_PPL.lvlibp:Father of all uServices.lvlib:Consumer Msg.lvclass:Do.vi:3750014->Actor Framework.lvlibp:Actor.lvclass:Receive Message.vi:1040017->Actor Framework.lvlibp:Actor.lvclass:Actor Core.vi:5880030->TALOS_PPL.lvlibp:Father of all uServices.lvlib:Father of all uServices.lvclass:Actor Core.vi:5880029->Actor Framework.lvlibp:Actor.lvclass:Actor.vi:6640035->Actor Framework.lvlibp:Actor.lvclass:Actor.vi.ACBRProxyCaller.28C00892"]


if __name__ == '__main__':
    filename = 'Error_Log_2024-05-29_13-52-03.842.txt'
    retry_datetimes = []
    with open(filename,'r') as f:
        for i, line in enumerate(f):
            error_code = re.search(r'(Code:) (5801|7105|7109|5218)',line) # retries are: banana returned retry, beam stopper in, empty shot, daq error
            if error_code is None:
                continue
            error_code = error_code.group()
            datetime_string = re.search(r'\d+:\d+:\d+\.\d+ \d+/\d+/\d+',line).group()
            retry_datetimes.append(datetime.strptime(datetime_string,r'%H:%M:%S.%f %d/%m/%Y'))
    
    for time in retry_datetimes:
        print(time)
    print(len(retry_datetimes))
    
