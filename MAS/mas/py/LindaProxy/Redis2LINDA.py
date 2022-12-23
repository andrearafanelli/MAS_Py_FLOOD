
'''
Copyright 2017-2018 Agnese Salutari.
Licensed under the Apache License, Version 2.0 (the "License"); 
you may not use this file except in compliance with the License. 
You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on 
an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. 
See the License for the specific language governing permissions and limitations under the License
'''

import LindaProxy.lindaproxy as lp
import redis

def makeAtomic(s):
    out = s.replace('(', 'A')
    out = out.replace(')', 'B')
    out = out.replace('[', 'C')
    out = out.replace(']', 'D')
    out = out.replace('.', 'E')
    out = out.replace(',', 'F')
    out = out.replace('/', 'G')
    out = out.replace('\\', 'H')
    out = out.replace("'", 'I')
    out = out.replace(' ', 'O')
    return out

# Canale di comunicazione da Redis a LINDA e da LINDA al MAS.
# Da usare come main rispetto a lindaproxy

#Il proxy è usato solo per inviare messaggi al MAS DALI
L = lp.LindaProxy(host='127.0.0.1')
L.connect()

#Il sistema inoltra gli eventi redis al MAS
R = redis.Redis()
pubsub = R.pubsub()
pubsub.subscribe('LINDAchannel')
for item in pubsub.listen():
    if item['type']=='message':
        msg=item['data'].decode('utf-8')
        atomic = makeAtomic(msg)
        print('evento redis', msg, atomic)
        L.send_message('masa', "redis(" + atomic + ")")
