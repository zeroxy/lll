!pip install joblib pyquery
import requests
import pyquery
from time import sleep

import tensorflow as tf
import datetime 
from joblib import Parallel, delayed, cpu_count
from itertools import permutations as comb
import numpy as np
np.set_printoptions(precision=8, suppress=True, linewidth=120, threshold=np.inf)
np.seterr(all="ignore")

def now():
    KST = datetime.timezone(datetime.timedelta(hours=9))
    return datetime.datetime.now(tz = KST)    

def correct_test(pb, bias, y_, testtime):
    count = np.zeros((pb.shape[0],7),dtype=np.int32)
    if pb.shape[1] !=45:
        print(f'pb : {pb.shape} s column size is not 45')
        return count
    pb = pb / np.sum(pb, axis=1).reshape((-1,1))
    pb += bias
    pb = pb / np.sum(pb, axis=1).reshape((-1,1))
    for i in range(pb.shape[0]):
        real = np.where(y_[i])[0]
        if np.sum(pb[i])== 0:
            pb[i] += 1
            pb[i] /= 45
        for f in range(testtime):
            virtual = np.random.choice(45,6, replace=False, p = pb[i])
            include = np.sum(np.in1d(virtual, real))
            count[i, include]+=1
    return count

cpucnt = cpu_count()
multipool = Parallel(n_jobs=(cpucnt*2), backend='multiprocessing')
poolfn = delayed(correct_test)

traindataset = 810
testtimes = 1000
epoch = 15
step = 2000
lr = 0.0002
combnum = 3
important_report= True


try:
    from google.colab import drive
    drive.mount('/content/gdrive')
    rootd='gdrive/My Drive/lotto/'
except ImportError:
    print("It's not google colab!!!!!")
    rootd='./'

ckptd=rootd+f'fc_multi_model_{combnum}/'
try:
    with np.load(rootd+'lottos_db.npz') as a :
        lottos = a['lottos']
        bonus = a['add_lottos']
except Error as e:
    lottos=np.zeros((0,45),dtype=np.int16)
    add_lottos = np.zeros((0,45),dtype=np.int16)

def get_url(no):
    return f'https://search.naver.com/search.naver?query={no+1}회로또'
    
KST = datetime.timezone(datetime.timedelta(hours=9))
firstgame_date = datetime.datetime(2002,12,7,21,00, tzinfo=KST)
gamedaydelta = now() - firstgame_date
if gamedaydelta.days/7 >= lottos.shape[0]:
    print("let's update!!")
    url = get_url(9998)
    body = requests.get(url)
    while body.status_code != 200:
        print(f'\r{body.status_code} {now()}', end='')
        sleep(5)
        body = requests.get(url)
    d = pyquery.PyQuery(body.text)('._lotto-btn-current em')
    limit = int(d.html()[:-1])
    end = lottos.shape[0]

    def get_balls(no):
        result = []
        url = get_url(no)
        body = requests.get(url)
        while True:
            d = pyquery.PyQuery(body.text)('.num_box .num')
            result = [ int(x.text)-1 for x in d]
            if len(result)==7:
                break
            sleep(5)
            body = requests.get(url)
        print(f'\r    {no+1} ', end='')
        if no%180 == 179:
            print(f'crawling...  {now()}')
        return result

    if limit>end:
        lottos = np.append(lottos,np.zeros((limit-end,45)),axis=0)
        add_lottos = np.append(add_lottos,np.zeros((limit-end,45)),axis=0)
        print(lottos.shape)
        verb=0
        crawled = multipool(
            delayed(get_balls)(x) for x in range(end,limit)
        )
        print(f'crawled!!  {now()}')
        for rowno,row in enumerate(crawled):
            lottos[end+rowno,row]=1
            add_lottos[end+rowno,row[-1]] = 1
        np.savez_compressed('lottos_db.npz', lottos=lottos, add_lottos=add_lottos)
        print(f'\nwe had {end}. so update to {limit}. now we have {lottos.shape[0]} rows.')
    else:
        print('\nno update')
    
data = lottos - bonus
neg_data = 1-data
cnted_lot = np.ones_like(neg_data)
cnted_lot[0] *= neg_data[0]
for  i in range(1, neg_data.shape[0]):
    cnted_lot[i] += cnted_lot[i-1]
    cnted_lot[i] *= neg_data[i]
print(f'{cnted_lot[:2,:30]}\n{data[:2,:30]}')
print(f'{cnted_lot[-1:,:30]}\n{data[-1:,:30]}')


#testidx = np.random.choice(data.shape[0]-1, data.shape[0]-1-traindataset, replace=False)
#trainidx = np.delete(np.arange(data.shape[0]-1),testidx)
trainidx = np.arange(traindataset)
testidx = np.arange(traindataset, data.shape[0]-1)

np.random.shuffle(trainidx)
np.random.shuffle(testidx)


money = np.asarray([0,0,0,5,50,1700,7000])
bias_sample = np.arange(0.000, 0.00002, 0.0001)

data_desc = f'train = {trainidx.shape}, test = {testidx.shape}, all data = {lottos.shape}'
model_desc = f'lr = {lr}, bias sample = {bias_sample}'
print(model_desc+'\n'+data_desc)

x = tf.placeholder(tf.float32, shape=(None, 45))
y = tf.placeholder(tf.float32, shape=(None, 45))

####### model setting ########
channels = [128,256,1024]
layer_type = []
for i in channels:
    layer_type.append(('fc',i,0.01))
    layer_type.append(('fc',i,0.))
layer_type.append(('dr',0.5,0.5))
layer_type.append(('dr',0.7,0.7))
print(layer_type)

nets = {}
def net(layer, is_training, reuse):
    
    for e in range(combnum):
        li = list(comb(layer_type,e))
        print(f'{e} ==> {len(li)}')
        for row in li:
            key = 'lot'+"_".join([f'{r[0]}{r[1]}-{r[2]}' for r in row])
            with tf.variable_scope(key, reuse=reuse):
                nets[key]= tf.layers.flatten(layer, name="flatten")
                for idx,(ltype,chan,rate) in enumerate(row):
                    if ltype=='fc':
                        if rate == 0.:
                            nets[key] = tf.layers.dense(nets[key], chan , activation=tf.nn.sigmoid, trainable=is_training, name=f"fc{idx}")
                        else:
                            nets[key] = tf.layers.dense(nets[key], chan , activation=tf.nn.sigmoid, trainable=is_training, name=f"fc{idx}",
                                                      kernel_regularizer=tf.contrib.layers.l2_regularizer(scale=rate))
                    elif ltype=='dr':
                        nets[key] = tf.layers.dropout(nets[key], rate=rate , training=is_training, name=f"dr{idx}")
                nets[key] = tf.layers.dense(nets[key], 45 , activation=tf.nn.sigmoid, trainable=is_training, name=f"fc_fin")
    return nets

trainnets = net(x, True, False)
testnets = net(x,False, True)
loss = {}
trainjob = {}
starttime = now()

for idx,k in enumerate(nets.keys()):
    loss[k] = tf.losses.mean_squared_error(y, trainnets[k]) + tf.losses.get_regularization_loss(scope=k, name=f'{k}_reg_loss')
    trainjob[k] = tf.train.RMSPropOptimizer(learning_rate=lr).minimize(loss[k])
    if idx%10 == 0:
        print(f'\r{idx} / {len(nets.keys())}   {now() - starttime}', end='')

config = tf.ConfigProto()
config.gpu_options.allow_growth = True
print("######")
with tf.Session(config=config) as sess:
    sess.run(tf.global_variables_initializer())
    saver = tf.train.Saver(keep_checkpoint_every_n_hours=0.7)
    ckptfile = tf.train.latest_checkpoint(ckptd)
    if ckptfile is not None:
        saver.restore(sess, ckptfile)
    resultlot=[]
    print(f'\n\n{now()}  -  {now() - starttime}')
    for jj in range(epoch):
        for ii in range(step):
            trainloss ,_ = sess.run([loss, trainjob], feed_dict={x:cnted_lot[trainidx], y:data[trainidx+1]})
            if ii % 50 == 0 :
                print(f'\r== {ii}==  {now()-starttime}',end='')
        saver.save(sess, ckptd+f'{lottos.shape[0]}_datas')
        print(f'  {now()} ')

        #### test phase!!!!! #####
        testresult,x_, y_ = sess.run([testnets,x, y], feed_dict={x:cnted_lot[testidx], y:data[testidx+1]})
        realtest,realx_, realy_ = sess.run([testnets,x,  y], feed_dict={x:cnted_lot[-1:], y:data[-1:]})
        print(f'== {jj+1} ==>>> {(realx_.astype(np.int32))[0,:5]}==  {now()-starttime}  {now()}')
        #count = np.zeros((len(nets.keys()),len(testidx),7),dtype=np.int32) 
        for pb_bias in bias_sample:
            returns = multipool(poolfn(testresult[k],pb_bias,y_, testtimes) for kidx,k in enumerate(nets.keys()))
            count = np.stack(returns)
            #for r in range(len(returns)):
            #    count[r] = returns[r]
            meancnt = np.mean(count, axis=1)
            report = ""
            bestreport = ""
            for kidx, k in enumerate(nets.keys()):
                tempreport = f'{k} == step : {jj+1}, pb bias :{pb_bias} ==>> {np.around(np.sum(meancnt[kidx, 3:]),2)}  '
                tempreport += f'{np.around(np.sum(meancnt[kidx]*money),2)} won / {testtimes}\n{np.around(meancnt[kidx],2)}\n'

                #### real test phase!!!!! #####

                if np.sum(meancnt[kidx]*money) >= (testtimes*0.7):
                    virtual = np.zeros((10,6),dtype=np.int32)
                    for f in range(virtual.shape[0]):
                        pb = realtest[k][0]
                        pb /= np.sum(pb)
                        pb += pb_bias
                        pb /= np.sum(pb)
                        virtual[f] = np.sort(np.random.choice(45,6, replace=False, p = pb))+1
                    tempvirt = np.unique(virtual, axis=0)
                    resultlot.append( tempvirt)
                    tempreport += f'{tempvirt}\n'
                    tempreport += f'{realtest[k][0] }\n\n'
                    bestreport += tempreport
                else :
                    report += tempreport
            if important_report:
                print(f'{bestreport} \n#############    {now()-starttime}  {now()}    #################\n')
            else:
                print(f'{report}\n\n{bestreport} \n#############    {now()-starttime}  {now()}    #################\n')
            if len(resultlot) > 0:
                tempresultlot = np.concatenate(resultlot)
                if tempresultlot.shape[0] <= 10:
                    continue
                tempresultidx = np.random.choice(tempresultlot.shape[0],10, replace=False)

                print('============= real luck!! =====================')
                print(f'{np.unique(tempresultlot[tempresultidx], axis=0)}')

                print('===============================================\n')
                print('============= luck in statistics! =============')

                luck_unique, luck_index, luck_count = np.unique(tempresultlot, axis=0, return_index=True,return_counts=True)

                luck_uniq_bin_cnt = np.bincount(luck_count)
                splitguide = luck_uniq_bin_cnt.shape[0]
                for i in range(1,splitguide):
                    if np.sum(luck_uniq_bin_cnt[-i:])>5:
                        splitguide-=i
                        splitguide-=1
                        print(i,splitguide)
                        break
                luck_uniq_row_cnt = luck_count[luck_count>splitguide]
                luck_uniq_row = tempresultlot[luck_index[luck_count>splitguide]]
                print(f'bincount : {luck_uniq_bin_cnt}')
                print(f'luckcount : {luck_uniq_row_cnt}')
                print(f'{luck_uniq_row}')
                print('===============================================')
                luck_top_five = np.random.choice(luck_uniq_row.shape[0], 5, replace=False)
                print(f'{luck_uniq_row[luck_top_five]}')
