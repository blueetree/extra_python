{\rtf1\ansi\ansicpg1252\cocoartf1671\cocoasubrtf100
{\fonttbl\f0\fnil\fcharset0 Monaco;}
{\colortbl;\red255\green255\blue255;\red38\green38\blue38;\red250\green249\blue246;}
{\*\expandedcolortbl;;\cssrgb\c20000\c20000\c20000;\cssrgb\c98431\c98039\c97255;}
\margl1440\margr1440\vieww10800\viewh8400\viewkind0
\deftab720
\pard\pardeftab720\sl380\partightenfactor0

\f0\fs28 \cf2 \cb3 \expnd0\expndtw0\kerning0
def balanceData(train_x_h, train_y_h, cntar_h):
\fs24 \cb1 \

\fs28 \cb3 \'a0\'a0\'a0\'a0L1 = train_x_h[cntar_h + "_lag_1"].tolist()
\fs24 \cb1 \

\fs28 \cb3 \'a0\'a0\'a0\'a0L2 = train_y_h.tolist()
\fs24 \cb1 \

\fs28 \cb3 \'a0\'a0\'a0\'a0nochangebool = [L1[xx] == L2[xx] for xx in range(len(L2))]
\fs24 \cb1 \

\fs28 \cb3 \'a0\'a0\'a0\'a0changeinds = [xx for xx in range(len(nochangebool)) if not nochangebool[xx]]
\fs24 \cb1 \

\fs28 \cb3 \'a0\'a0\'a0\'a0nochangeinds = [xx for xx in range(len(nochangebool)) if nochangebool[xx]]
\fs24 \cb1 \

\fs28 \cb3 \'a0\'a0\'a0\'a0nochangess = np.random.choice(nochangeinds, len(changeinds), replace=False).tolist()
\fs24 \cb1 \

\fs28 \cb3 \'a0\'a0\'a0\'a0nki = nochangess + changeinds
\fs24 \cb1 \

\fs28 \cb3 \'a0\'a0\'a0\'a0train_x_f = train_x_h.iloc[nki]
\fs24 \cb1 \

\fs28 \cb3 \'a0\'a0\'a0\'a0train_y_f = train_y_h.iloc[nki]
\fs24 \cb1 \

\fs28 \cb3 \'a0\'a0\'a0\'a0return train_x_f, train_y_f, nki
\fs24 \cb1 \
\pard\pardeftab720\sl380\partightenfactor0

\fs28 \cf2 \
\
\pard\pardeftab720\sl380\partightenfactor0
\cf2 \cb3 def FilterFeatures(train_x_h,train_y_h,test_x_h,FSparams_h):
\fs24 \cb1 \

\fs28 \cb3 \'a0\'a0\'a0\'a0if FSparams_h['do_FS']:
\fs24 \cb1 \

\fs28 \cb3 \'a0\'a0\'a0\'a0\'a0\'a0\'a0\'a0FI_order = rank_features(train_x_h, train_y_h, method=FSparams_h['FSmethod'], continuous=True)
\fs24 \cb1 \

\fs28 \cb3 \'a0\'a0\'a0\'a0\'a0\'a0\'a0\'a0SelectedFeatures_loc = FI_order[:FSparams_h['NSF']]
\fs24 \cb1 \

\fs28 \cb3 \'a0\'a0\'a0\'a0\'a0\'a0\'a0\'a0train_x_f = train_x_h.iloc[:,SelectedFeatures_loc]
\fs24 \cb1 \

\fs28 \cb3 \'a0\'a0\'a0\'a0\'a0\'a0\'a0\'a0test_x_f = test_x_h.iloc[:,SelectedFeatures_loc]
\fs24 \cb1 \

\fs28 \cb3 \'a0\'a0\'a0\'a0else:
\fs24 \cb1 \

\fs28 \cb3 \'a0\'a0\'a0\'a0\'a0\'a0\'a0\'a0train_x_f = train_x_h
\fs24 \cb1 \

\fs28 \cb3 \'a0\'a0\'a0\'a0\'a0\'a0\'a0\'a0test_x_f = test_x_h
\fs24 \cb1 \

\fs28 \cb3 \'a0\'a0\'a0\'a0\'a0\'a0\'a0\'a0SelectedFeatures_loc = -1
\fs24 \cb1 \

\fs28 \cb3 \'a0\'a0\'a0\'a0return train_x_f, test_x_f, SelectedFeatures_loc
\fs24 \cb1 \
\pard\pardeftab720\sl380\partightenfactor0

\fs28 \cf2 \
\
\
\pard\pardeftab720\sl380\partightenfactor0
\cf2 \cb3 def fsloopsfromdict(FSparamdict_h):
\fs24 \cb1 \

\fs28 \cb3 \'a0\'a0\'a0\'a0FSparamsLooplist_h = []
\fs24 \cb1 \

\fs28 \cb3 \'a0\'a0\'a0\'a0if False in FSparamdict_h['do_FS']:
\fs24 \cb1 \

\fs28 \cb3 \'a0\'a0\'a0\'a0\'a0\'a0\'a0\'a0FSparamsLooplist_h = FSparamsLooplist_h + [\{'do_FS': False,'FSmethod': 'NoFS','NSF': 0\}]
\fs24 \cb1 \

\fs28 \cb3 \'a0\'a0\'a0\'a0if True in FSparamdict_h['do_FS']:
\fs24 \cb1 \

\fs28 \cb3 \'a0\'a0\'a0\'a0\'a0\'a0\'a0\'a0IL =list(itertools.product([True],FSparamdict_h['FSmethod'],FSparamdict_h['NSF']))
\fs24 \cb1 \

\fs28 \cb3 \'a0\'a0\'a0\'a0\'a0\'a0\'a0\'a0FSparamsLooplist_h = FSparamsLooplist_h + [\{'do_FS': xx[0],'FSmethod': xx[1],'NSF': xx[2]\} for xx in IL]
\fs24 \cb1 \

\fs28 \cb3 \'a0\'a0\'a0\'a0return FSparamsLooplist_h
\fs24 \cb1 \
\pard\pardeftab720\sl380\partightenfactor0

\fs28 \cf2 \
\
\
\pard\pardeftab720\sl380\partightenfactor0
\cf2 \cb3 def AssemblePreds(FullPredsDict_l):
\fs24 \cb1 \

\fs28 \cb3 \'a0\'a0\'a0\'a0FullPredsDict_l['FullPredictionsAssembled']=\{\}
\fs24 \cb1 \

\fs28 \cb3 \'a0\'a0\'a0\'a0foldind = FullPredsDict_l['FullPredictionsKeyMeaning'].index('Fold')
\fs24 \cb1 \

\fs28 \cb3 \'a0\'a0\'a0\'a0fpk = [make_tuple(xx) for xx in FullPredsDict_l['FullPredictions'].keys()]
\fs24 \cb1 \

\fs28 \cb3 \'a0\'a0\'a0\'a0fpkl = [list(xx) for xx in fpk]
\fs24 \cb1 \

\fs28 \cb3 \'a0\'a0\'a0\'a0fpklm = [xx[:foldind] + xx[foldind + 1:] for xx in fpkl]
\fs24 \cb1 \

\fs28 \cb3 \'a0\'a0\'a0\'a0fpklmu = [list(i) for i in set(tuple(i) for i in fpklm)]
\fs24 \cb1 \

\fs28 \cb3 \'a0\'a0\'a0\'a0nwof = FullPredsDict_l['FullPredictionsKeyMeaning'][:foldind] + \\
\fs24 \cb1 \

\fs28 \cb3 \'a0\'a0\'a0\'a0FullPredsDict_l['FullPredictionsKeyMeaning'][foldind +1:]
\fs24 \cb1 \

\fs28 \cb3 \'a0\'a0\'a0\'a0FullPredsDict_l['FullPredictionsAssembledKeyMeaning'] = nwof
\fs24 \cb1 \

\fs28 \cb3 \'a0\'a0\'a0\'a0newrepind = nwof.index('Rep')
\fs24 \cb1 \

\fs28 \cb3 \'a0\'a0\'a0\'a0for ml in fpklmu:
\fs24 \cb1 \

\fs28 \cb3 \'a0\'a0\'a0\'a0\'a0\'a0\'a0\'a0FullPredsDict_l['FullPredictionsAssembled'][tuple(ml)]=np.full(len(FullPredsDict_l['Target']),np.nan)
\fs24 \cb1 \

\fs28 \cb3 \'a0\'a0\'a0\'a0\'a0\'a0\'a0\'a0for fn in range(FullPredsDict_l['K']):
\fs24 \cb1 \

\fs28 \cb3 \'a0\'a0\'a0\'a0\'a0\'a0\'a0\'a0\'a0\'a0\'a0\'a0predinds = FullPredsDict_l['RepFoldIndices'][ml[newrepind]][fn]
\fs24 \cb1 \

\fs28 \cb3 \'a0\'a0\'a0\'a0\'a0\'a0\'a0\'a0\'a0\'a0\'a0\'a0fullkey = ml[:]
\fs24 \cb1 \

\fs28 \cb3 \'a0\'a0\'a0\'a0\'a0\'a0\'a0\'a0\'a0\'a0\'a0\'a0fullkey.insert(foldind,fn)
\fs24 \cb1 \

\fs28 \cb3 \'a0\'a0\'a0\'a0\'a0\'a0\'a0\'a0\'a0\'a0\'a0\'a0FullPredsDict_l['FullPredictionsAssembled'][tuple(ml)][predinds] = FullPredsDict_l['FullPredictions'][str(tuple(fullkey))]
\fs24 \cb1 \

\fs28 \cb3 \'a0\'a0\'a0\'a0return FullPredsDict_l
\fs24 \cb1 \
\pard\pardeftab720\sl380\partightenfactor0

\fs28 \cf2 \
\pard\pardeftab720\sl380\partightenfactor0
\cf2 \cb3 def CalcMSEs(FullPredsDict_l):
\fs24 \cb1 \

\fs28 \cb3 \'a0\'a0\'a0\'a0FullPredsDict_l['MSE']=\{\}
\fs24 \cb1 \

\fs28 \cb3 \'a0\'a0\'a0\'a0for thiskey in FullPredsDict_l['FullPredictionsAssembled'].keys():
\fs24 \cb1 \

\fs28 \cb3 \'a0\'a0\'a0\'a0\'a0\'a0\'a0\'a0preds = FullPredsDict_l['FullPredictionsAssembled'][thiskey]
\fs24 \cb1 \

\fs28 \cb3 \'a0\'a0\'a0\'a0\'a0\'a0\'a0\'a0FullPredsDict_l['MSE'][thiskey] = mean_squared_error(preds, FullPredsDict_l['Target']) # this was Tar..
\fs24 \cb1 \

\fs28 \cb3 \'a0\'a0\'a0\'a0return FullPredsDict_l
\fs24 \cb1 \
\pard\pardeftab720\sl380\partightenfactor0

\fs28 \cf2 \
\
\
\pard\pardeftab720\sl380\partightenfactor0
\cf2 \cb3 def CalcMSE_mean_std(FullPredsDict_l):
\fs24 \cb1 \

\fs28 \cb3 \'a0\'a0\'a0\'a0msekeys = FullPredsDict_l['MSE'].keys()
\fs24 \cb1 \

\fs28 \cb3 \'a0\'a0\'a0\'a0repind = FullPredsDict_l['FullPredictionsAssembledKeyMeaning'].index("Rep")
\fs24 \cb1 \

\fs28 \cb3 \'a0\'a0\'a0\'a0fpkl = [list(xx) for xx in msekeys]
\fs24 \cb1 \

\fs28 \cb3 \'a0\'a0\'a0\'a0fpklm = [xx[:repind] + xx[repind + 1:] for xx in fpkl]
\fs24 \cb1 \

\fs28 \cb3 \'a0\'a0\'a0\'a0fpklmu = [list(i) for i in set(tuple(i) for i in fpklm)]
\fs24 \cb1 \

\fs28 \cb3 \'a0\'a0\'a0\'a0FullPredsDict_l['MSE_mean']=\{\}
\fs24 \cb1 \

\fs28 \cb3 \'a0\'a0\'a0\'a0FullPredsDict_l['MSE_std']=\{\}
\fs24 \cb1 \

\fs28 \cb3 \'a0\'a0\'a0\'a0for thiskey in fpklmu:
\fs24 \cb1 \

\fs28 \cb3 \'a0\'a0\'a0\'a0\'a0\'a0\'a0\'a0tkc = thiskey[:]
\fs24 \cb1 \

\fs28 \cb3 \'a0\'a0\'a0\'a0\'a0\'a0\'a0\'a0thesemses = []
\fs24 \cb1 \

\fs28 \cb3 \'a0\'a0\'a0\'a0\'a0\'a0\'a0\'a0for rep in range(FullPredsDict_l["Reps"]):
\fs24 \cb1 \

\fs28 \cb3 \'a0\'a0\'a0\'a0\'a0\'a0\'a0\'a0\'a0\'a0\'a0\'a0trk = tkc[:]
\fs24 \cb1 \

\fs28 \cb3 \'a0\'a0\'a0\'a0\'a0\'a0\'a0\'a0\'a0\'a0\'a0\'a0trk.insert(repind,rep)
\fs24 \cb1 \

\fs28 \cb3 \'a0\'a0\'a0\'a0\'a0\'a0\'a0\'a0\'a0\'a0\'a0\'a0thesemses.append(FullPredsDict_l['MSE'][tuple(trk)])
\fs24 \cb1 \

\fs28 \cb3 \'a0\'a0\'a0\'a0\'a0\'a0\'a0\'a0FullPredsDict_l['MSE_mean'][tuple(thiskey)] = np.mean(thesemses)
\fs24 \cb1 \

\fs28 \cb3 \'a0\'a0\'a0\'a0\'a0\'a0\'a0\'a0FullPredsDict_l['MSE_std'][tuple(thiskey)] = np.std(thesemses)
\fs24 \cb1 \

\fs28 \cb3 \'a0\'a0\'a0\'a0\'a0\'a0\'a0\'a0FullPredsDict_l['MSE_mean_std_KeyMeaning'] =[xx for xx in FullPredsDict_l['FullPredictionsAssembledKeyMeaning'] if xx != "Rep"]
\fs24 \cb1 \

\fs28 \cb3 \'a0\'a0\'a0\'a0return FullPredsDict_l
\fs24 \cb1 \
\pard\pardeftab720\sl380\partightenfactor0

\fs28 \cf2 \
\
\
\pard\pardeftab720\sl380\partightenfactor0
\cf2 \cb3 def mergeFullPreds(fpd1_in,fpd2):
\fs24 \cb1 \

\fs28 \cb3 \'a0\'a0\'a0\'a0fpd1 = copy.deepcopy(fpd1_in)
\fs24 \cb1 \

\fs28 \cb3 \'a0\'a0\'a0\'a0fpd1['SelectedFeatures'].update(fpd2['SelectedFeatures'])
\fs24 \cb1 \

\fs28 \cb3 \'a0\'a0\'a0\'a0fpd1['FullPredictions'].update(fpd2['FullPredictions'])
\fs24 \cb1 \

\fs28 \cb3 \'a0\'a0\'a0\'a0return fpd1
\fs24 \cb1 \
\pard\pardeftab720\sl380\partightenfactor0

\fs28 \cf2 \
\
\pard\pardeftab720\sl380\partightenfactor0
\cf2 \cb3 def PredictionFeedback(mod,dat,Target_loc,numlags):
\fs24 \cb1 \

\fs28 \cb3 \'a0\'a0\'a0\'a0adjustcols = [Target_loc.split("_")[0] + "_lag_" + str(xx) for xx in range(1, numlags)]
\fs24 \cb1 \

\fs28 \cb3 \'a0\'a0\'a0\'a0PredictionsList = []
\fs24 \cb1 \

\fs28 \cb3 \'a0\'a0\'a0\'a0ds0 = dat.shape[0]
\fs24 \cb1 \

\fs28 \cb3 \'a0\'a0\'a0\'a0bufferd = deque(dat.iloc[[0]][adjustcols].values[0])
\fs24 \cb1 \

\fs28 \cb3 \'a0\'a0\'a0\'a0for sampnum in range(ds0):
\fs24 \cb1 \

\fs28 \cb3 \'a0\'a0\'a0\'a0\'a0\'a0\'a0\'a0x1 = dat.iloc[[sampnum]].copy()
\fs24 \cb1 \

\fs28 \cb3 \'a0\'a0\'a0\'a0\'a0\'a0\'a0\'a0\'a0x1.loc[x1.index[0], adjustcols] = bufferd
\fs24 \cb1 \

\fs28 \cb3 \'a0\'a0\'a0\'a0\'a0\'a0\'a0\'a0Pred1 = mod.predict(x1)
\fs24 \cb1 \

\fs28 \cb3 \'a0\'a0\'a0\'a0\'a0\'a0\'a0\'a0bufferd.appendleft(Pred1[0])
\fs24 \cb1 \

\fs28 \cb3 \'a0\'a0\'a0\'a0\'a0\'a0\'a0\'a0bufferd.pop()
\fs24 \cb1 \

\fs28 \cb3 \'a0\'a0\'a0\'a0\'a0\'a0\'a0\'a0PredictionsList.append(Pred1[0])
\fs24 \cb1 \

\fs28 \cb3 \'a0\'a0\'a0\'a0return PredictionsList
\fs24 \cb1 \
\pard\pardeftab720\sl380\partightenfactor0

\fs28 \cf2 \
\
\pard\pardeftab720\sl380\partightenfactor0
\cf2 \cb3 def CheckTimestampOrder(dat):
\fs24 \cb1 \

\fs28 \cb3 \'a0\'a0\'a0\'a0tsdt = pd.to_datetime(dat['timestamp'], format='%Y-%m-%d %H:%M:%S')
\fs24 \cb1 \

\fs28 \cb3 \'a0\'a0\'a0\'a0return tsdt.tolist() == sorted(tsdt.tolist())
\fs24 \cb1 \
\pard\pardeftab720\sl380\partightenfactor0

\fs28 \cf2 \
\
\pard\pardeftab720\sl380\partightenfactor0
\cf2 \cb3 # take a look at the two sets of predictions vs true
\fs24 \cb1 \

\fs28 \cb3 def FullPlot(winpreds_h,Tar_h,ptss_h):
\fs24 \cb1 \

\fs28 \cb3 \'a0\'a0\'a0\'a0fig,axarr = plt.subplots(1,sharex=True,figsize=(10, 1))
\fs24 \cb1 \

\fs28 \cb3 \'a0\'a0\'a0\'a0axarr.plot(winpreds_h[0:ptss_h], color='blue')
\fs24 \cb1 \

\fs28 \cb3 \'a0\'a0\'a0\'a0axarr.plot(Tar_h[0:ptss_h], color='red')
\fs24 \cb1 \

\fs28 \cb3 \'a0\'a0\'a0\'a0box = axarr.get_position()
\fs24 \cb1 \

\fs28 \cb3 \'a0\'a0\'a0\'a0axarr.set_position([box.x0, box.y0 +0.1, box.width * 0.8, box.height * 0.7])
\fs24 \cb1 \

\fs28 \cb3 \'a0\'a0\'a0\'a0axarr.legend(['Predicted', 'True Value'], loc='center left', bbox_to_anchor=(1, 0.5))
\fs24 \cb1 \

\fs28 \cb3 \'a0\'a0\'a0\'a0return plt
\fs24 \cb1 \
\pard\pardeftab720\sl380\partightenfactor0

\fs28 \cf2 \
\
\
\pard\pardeftab720\sl380\partightenfactor0
\cf2 \cb3 def FullPlot_ax(winpreds_h,Tar_h,ptss_h,axarr, ttl=""):
\fs24 \cb1 \

\fs28 \cb3 \'a0\'a0\'a0\'a0# fig,axarr = plt.subplots(1,sharex=True,figsize=(10, 1))
\fs24 \cb1 \

\fs28 \cb3 \'a0\'a0\'a0\'a0axarr.plot(winpreds_h[0:ptss_h], color='blue')
\fs24 \cb1 \

\fs28 \cb3 \'a0\'a0\'a0\'a0axarr.plot(Tar_h[0:ptss_h], color='red')
\fs24 \cb1 \

\fs28 \cb3 \'a0\'a0\'a0\'a0box = axarr.get_position()
\fs24 \cb1 \

\fs28 \cb3 \'a0\'a0\'a0\'a0axarr.set_position([box.x0, box.y0 +0.1, box.width * 0.8, box.height * 0.7])
\fs24 \cb1 \

\fs28 \cb3 \'a0\'a0\'a0\'a0axarr.legend(['Predicted', 'True Value'], loc='center left', bbox_to_anchor=(1, 0.5))
\fs24 \cb1 \

\fs28 \cb3 \'a0\'a0\'a0\'a0axarr.set_title(ttl)
\fs24 \cb1 \

\fs28 \cb3 \'a0\'a0\'a0\'a0return plt
\fs24 \cb1 \
\pard\pardeftab720\sl380\partightenfactor0

\fs28 \cf2 \
\
\
\pard\pardeftab720\sl380\partightenfactor0
\cf2 \cb3 def FeaturePlot_ax(TripData_h,ptss_h,axarr,ttl="Environment"):
\fs24 \cb1 \

\fs28 \cb3 \'a0\'a0\'a0\'a0colset = cm.rainbow(np.linspace(0, 1, 8))
\fs24 \cb1 \

\fs28 \cb3 \'a0\'a0\'a0\'a0axarr.plot((TripData_h['OtsAirTmpCrVal_lag_1'][0:ptss_h]/max(TripData_h['OtsAirTmpCrVal_lag_1'])).tolist(), color = colset[0])
\fs24 \cb1 \

\fs28 \cb3 \'a0\'a0\'a0\'a0axarr.plot((TripData_h['IPSnsrSolrInt_lag_1'][0:ptss_h]/max(TripData_h['IPSnsrSolrInt_lag_1'])).tolist(), color = colset[1])
\fs24 \cb1 \

\fs28 \cb3 \'a0\'a0\'a0\'a0axarr.plot((TripData_h['EngSpd_lag_1'][0:ptss_h]/max(TripData_h['EngSpd_lag_1'])).tolist(), color = colset[2])
\fs24 \cb1 \

\fs28 \cb3 \'a0\'a0\'a0\'a0axarr.plot((TripData_h['DriverSetTemp_lag_1'][0:ptss_h]/max(TripData_h['DriverSetTemp_lag_1'])).tolist(), color = colset[3])
\fs24 \cb1 \

\fs28 \cb3 \'a0\'a0\'a0\'a0axarr.plot((TripData_h['WindPattern_lag_1'][0:ptss_h]).tolist(), color = colset[4])
\fs24 \cb1 \

\fs28 \cb3 \'a0\'a0\'a0\'a0axarr.plot((TripData_h['WindLevel_lag_1'][0:ptss_h]).tolist(), color = colset[5])
\fs24 \cb1 \

\fs28 \cb3 \'a0\'a0\'a0\'a0axarr.plot((TripData_h['LftLoDctTemp_lag_1'][0:ptss_h]/max(TripData_h['LftLoDctTemp_lag_1'])).tolist(), color = colset[6])
\fs24 \cb1 \

\fs28 \cb3 \'a0\'a0\'a0\'a0axarr.plot((TripData_h['LftUpDctTemp_lag_1'][0:ptss_h]/max(TripData_h['LftUpDctTemp_lag_1'])).tolist(), color = colset[7])
\fs24 \cb1 \

\fs28 \cb3 \'a0\'a0\'a0\'a0#axarr.plot(Tar_h[0:ptss_h], color='red')
\fs24 \cb1 \

\fs28 \cb3 \'a0\'a0\'a0\'a0box = axarr.get_position()
\fs24 \cb1 \

\fs28 \cb3 \'a0\'a0\'a0\'a0axarr.set_position([box.x0, box.y0 +0.1, box.width * 0.8, box.height * 0.7])
\fs24 \cb1 \

\fs28 \cb3 \'a0\'a0\'a0\'a0axarr.legend(['OtsAirTmpCrVal_lag_1', 'IPSnsrSolrInt_lag_1', 'EngSpd_lag_1', 'DriverSetTemp_lag_1', 'WindPattern_lag_1', 'WindLevel_lag_1','LftLoDctTemp_lag_1','LftUpDctTemp_lag_1'],
\fs24 \cb1 \

\fs28 \cb3 \'a0\'a0\'a0\'a0loc='center left',
\fs24 \cb1 \

\fs28 \cb3 \'a0\'a0\'a0\'a0fontsize=4,
\fs24 \cb1 \

\fs28 \cb3 \'a0\'a0\'a0\'a0bbox_to_anchor=(1, 0.5))
\fs24 \cb1 \

\fs28 \cb3 \'a0\'a0\'a0\'a0axarr.set_title(ttl)
\fs24 \cb1 \

\fs28 \cb3 \'a0\'a0\'a0\'a0return plt
\fs24 \cb1 \
\pard\pardeftab720\sl380\partightenfactor0

\fs28 \cf2 \
\
\pard\pardeftab720\sl380\partightenfactor0
\cf2 \cb3 #########################
\fs24 \cb1 \

\fs28 \cb3 # 6
\fs24 \cb1 \

\fs28 \cb3 #########################
\fs24 \cb1 \

\fs28 \cb3 def stratified_cross_validation_splits(yy, K):
\fs24 \cb1 \

\fs28 \cb3 \'a0\'a0\'a0\'a0"""
\fs24 \cb1 \

\fs28 \cb3 \'a0\'a0\'a0\'a0Return a list of K lists (of approximately equal length) of indices of the input (yy)
\fs24 \cb1 \

\fs28 \cb3 \'a0\'a0\'a0\'a0where each list has approximately the same distribution of yy values as yy itself.
\fs24 \cb1 \
\pard\pardeftab720\sl380\partightenfactor0

\fs28 \cf2 \
\pard\pardeftab720\sl380\partightenfactor0
\cf2 \cb3 \'a0\'a0\'a0\'a0:param yy:\'a0\'a0A pandas series or a list.\'a0\'a0The target values (floats or integers) to be stratified and subdivided.
\fs24 \cb1 \

\fs28 \cb3 \'a0\'a0\'a0\'a0:param K: An integer.\'a0\'a0The number of partitions of the targeet.
\fs24 \cb1 \

\fs28 \cb3 \'a0\'a0\'a0\'a0:return: A list of lists of indices.
\fs24 \cb1 \

\fs28 \cb3 \'a0\'a0\'a0\'a0"""
\fs24 \cb1 \

\fs28 \cb3 \'a0\'a0\'a0\'a0# implement stratified cross validation splits
\fs24 \cb1 \

\fs28 \cb3 \'a0\'a0\'a0\'a0#
\fs24 \cb1 \

\fs28 \cb3 \'a0\'a0\'a0\'a0# here's a simple test that this works:
\fs24 \cb1 \

\fs28 \cb3 \'a0\'a0\'a0\'a0# Tar = np.random.choice(10,1000)
\fs24 \cb1 \

\fs28 \cb3 \'a0\'a0\'a0\'a0# Tdf = pd.DataFrame(\{'col':Tar\})
\fs24 \cb1 \

\fs28 \cb3 \'a0\'a0\'a0\'a0# tmp = sss(Tar.tolist(),5)
\fs24 \cb1 \

\fs28 \cb3 \'a0\'a0\'a0\'a0# tmp2 = sss(Tdf['col'],5)
\fs24 \cb1 \

\fs28 \cb3 \'a0\'a0\'a0\'a0# IS = np.zeros((5,5))
\fs24 \cb1 \

\fs28 \cb3 \'a0\'a0\'a0\'a0# IS2 = np.zeros((5,5))
\fs24 \cb1 \

\fs28 \cb3 \'a0\'a0\'a0\'a0# for i in range(len(tmp)):
\fs24 \cb1 \

\fs28 \cb3 \'a0\'a0\'a0\'a0#\'a0\'a0\'a0\'a0\'a0for j in range(len(tmp)):
\fs24 \cb1 \

\fs28 \cb3 \'a0\'a0\'a0\'a0#\'a0\'a0\'a0\'a0\'a0\'a0\'a0\'a0\'a0IS[i][j] = len(set(tmp[i]).intersection(set(tmp[j])))
\fs24 \cb1 \

\fs28 \cb3 \'a0\'a0\'a0\'a0#\'a0\'a0\'a0\'a0\'a0\'a0\'a0\'a0\'a0IS2[i][j] = len(set(tmp2[i]).intersection(set(tmp2[j])))
\fs24 \cb1 \

\fs28 \cb3 \'a0\'a0\'a0\'a0outinds = [[] for x in range(K)]
\fs24 \cb1 \

\fs28 \cb3 \'a0\'a0\'a0\'a0if isinstance(yy,pd.core.series.Series):
\fs24 \cb1 \

\fs28 \cb3 \'a0\'a0\'a0\'a0\'a0\'a0\'a0\'a0y = yy.tolist()
\fs24 \cb1 \

\fs28 \cb3 \'a0\'a0\'a0\'a0else:
\fs24 \cb1 \

\fs28 \cb3 \'a0\'a0\'a0\'a0\'a0\'a0\'a0\'a0if not isinstance(yy,list):
\fs24 \cb1 \

\fs28 \cb3 \'a0\'a0\'a0\'a0\'a0\'a0\'a0\'a0\'a0\'a0\'a0\'a0raise ValueError('Target values must be either a list or a pandas series')
\fs24 \cb1 \

\fs28 \cb3 \'a0\'a0\'a0\'a0\'a0\'a0\'a0\'a0y=yy[:]
\fs24 \cb1 \

\fs28 \cb3 \'a0\'a0\'a0\'a0ysi = np.argsort(y).tolist()
\fs24 \cb1 \

\fs28 \cb3 \'a0\'a0\'a0\'a0bi = range(0,len(y),K)
\fs24 \cb1 \

\fs28 \cb3 \'a0\'a0\'a0\'a0if len(y) not in bi:
\fs24 \cb1 \

\fs28 \cb3 \'a0\'a0\'a0\'a0\'a0\'a0\'a0\'a0bi = bi +[len(y)]
\fs24 \cb1 \

\fs28 \cb3 \'a0\'a0\'a0\'a0bir =[range(bi[x],bi[x+1]) for x in range(len(bi)-1)]
\fs24 \cb1 \

\fs28 \cb3 \'a0\'a0\'a0\'a0for ii in range(len(bir)):
\fs24 \cb1 \

\fs28 \cb3 \'a0\'a0\'a0\'a0\'a0\'a0\'a0\'a0random.shuffle(bir[ii])
\fs24 \cb1 \

\fs28 \cb3 \'a0\'a0\'a0\'a0for bs in range(len(bir)):
\fs24 \cb1 \

\fs28 \cb3 \'a0\'a0\'a0\'a0\'a0\'a0\'a0\'a0for bss in range(len(bir[bs])):
\fs24 \cb1 \

\fs28 \cb3 \'a0\'a0\'a0\'a0\'a0\'a0\'a0\'a0\'a0\'a0\'a0\'a0outinds[bss]=outinds[bss] + [ysi[bir[bs].pop()]]
\fs24 \cb1 \

\fs28 \cb3 \'a0\'a0\'a0\'a0return outinds
\fs24 \cb1 \
\pard\pardeftab720\sl380\partightenfactor0

\fs28 \cf2 \
\pard\pardeftab720\sl380\partightenfactor0
\cf2 \cb3 def check_make_path(pn):
\fs24 \cb1 \

\fs28 \cb3 \'a0\'a0\'a0\'a0if not os.path.exists(pn):
\fs24 \cb1 \

\fs28 \cb3 \'a0\'a0\'a0\'a0\'a0\'a0\'a0\'a0os.makedirs(pn)
\fs24 \cb1 \
\pard\pardeftab720\sl380\partightenfactor0

\fs28 \cf2 \
\
\
\pard\pardeftab720\sl380\partightenfactor0
\cf2 \cb3 def getTrainTest(data,targets,testinds):
\fs24 \cb1 \

\fs28 \cb3 \'a0\'a0\'a0\'a0traininds = [x for x in range(data.shape[0]) if x not in testinds]
\fs24 \cb1 \

\fs28 \cb3 \'a0\'a0\'a0\'a0train_x = data.iloc[traininds].reset_index(drop=True)
\fs24 \cb1 \

\fs28 \cb3 \'a0\'a0\'a0\'a0test_x = data.iloc[testinds].reset_index(drop=True)
\fs24 \cb1 \

\fs28 \cb3 \'a0\'a0\'a0\'a0train_y = targets.iloc[traininds].reset_index(drop=True)
\fs24 \cb1 \

\fs28 \cb3 \'a0\'a0\'a0\'a0test_y = targets.iloc[testinds].reset_index(drop=True)
\fs24 \cb1 \

\fs28 \cb3 \'a0\'a0\'a0\'a0return train_x, train_y, test_x, test_y
\fs24 \cb1 \
\pard\pardeftab720\sl380\partightenfactor0

\fs28 \cf2 \
\
\
\pard\pardeftab720\sl380\partightenfactor0
\cf2 \cb3 # implement stratified cross validation splits
\fs24 \cb1 \

\fs28 \cb3 def sss(yy,K):
\fs24 \cb1 \

\fs28 \cb3 \'a0\'a0\'a0\'a0outinds = [[] for x in range(K)]
\fs24 \cb1 \

\fs28 \cb3 \'a0\'a0\'a0\'a0if isinstance(yy,pd.core.series.Series):
\fs24 \cb1 \

\fs28 \cb3 \'a0\'a0\'a0\'a0\'a0\'a0\'a0\'a0y = yy.tolist()
\fs24 \cb1 \

\fs28 \cb3 \'a0\'a0\'a0\'a0else:
\fs24 \cb1 \

\fs28 \cb3 \'a0\'a0\'a0\'a0\'a0\'a0\'a0\'a0if not isinstance(yy,list):
\fs24 \cb1 \

\fs28 \cb3 \'a0\'a0\'a0\'a0\'a0\'a0\'a0\'a0\'a0\'a0\'a0\'a0raise ValueError('Target values must be either a list or a pandas series')
\fs24 \cb1 \

\fs28 \cb3 \'a0\'a0\'a0\'a0\'a0\'a0\'a0\'a0y=yy[:]
\fs24 \cb1 \

\fs28 \cb3 \'a0\'a0\'a0\'a0ysi = np.argsort(y).tolist()
\fs24 \cb1 \

\fs28 \cb3 \'a0\'a0\'a0\'a0bi = range(0,len(y),K)
\fs24 \cb1 \

\fs28 \cb3 \'a0\'a0\'a0\'a0if len(y) not in bi:
\fs24 \cb1 \

\fs28 \cb3 \'a0\'a0\'a0\'a0\'a0\'a0\'a0\'a0bi = bi +[len(y)]
\fs24 \cb1 \

\fs28 \cb3 \'a0\'a0\'a0\'a0bir =[range(bi[x],bi[x+1]) for x in range(len(bi)-1)]
\fs24 \cb1 \

\fs28 \cb3 \'a0\'a0\'a0\'a0for ii in range(len(bir)):
\fs24 \cb1 \

\fs28 \cb3 \'a0\'a0\'a0\'a0\'a0\'a0\'a0\'a0random.shuffle(bir[ii])
\fs24 \cb1 \

\fs28 \cb3 \'a0\'a0\'a0\'a0for bs in range(len(bir)):
\fs24 \cb1 \

\fs28 \cb3 \'a0\'a0\'a0\'a0\'a0\'a0\'a0\'a0for bss in range(len(bir[bs])):
\fs24 \cb1 \

\fs28 \cb3 \'a0\'a0\'a0\'a0\'a0\'a0\'a0\'a0\'a0\'a0\'a0\'a0outinds[bss]=outinds[bss] + [ysi[bir[bs].pop()]]
\fs24 \cb1 \

\fs28 \cb3 \'a0\'a0\'a0\'a0return outinds
\fs24 \cb1 \
\pard\pardeftab720\sl380\partightenfactor0

\fs28 \cf2 \
\
\
\pard\pardeftab720\sl380\partightenfactor0
\cf2 \cb3 def makeFV(numlag, rid, Data,LFloc,NLFloc):
\fs24 \cb1 \

\fs28 \cb3 \'a0\'a0\'a0\'a0if rid < numlag:
\fs24 \cb1 \

\fs28 \cb3 \'a0\'a0\'a0\'a0raise ValueError('Row number cannot be less than the lag')
\fs24 \cb1 \

\fs28 \cb3 \'a0\'a0\'a0\'a0# this is assuming the rows are in temporal order - ensure this earlier
\fs24 \cb1 \

\fs28 \cb3 \'a0\'a0\'a0\'a0FV = Data[NLFloc].iloc[rid]
\fs24 \cb1 \

\fs28 \cb3 \'a0\'a0\'a0\'a0LFm = Data[LFloc].iloc[(rid-numlag+1):(rid+1)][::-1]
\fs24 \cb1 \

\fs28 \cb3 \'a0\'a0\'a0\'a0LFv = flatten(LFm.as_matrix().tolist())
\fs24 \cb1 \

\fs28 \cb3 \'a0\'a0\'a0\'a0featvec = FV.tolist() + LFv
\fs24 \cb1 \

\fs28 \cb3 \'a0\'a0\'a0\'a0featnames = list(FV.index) + [x+"_lag_" + str(y) for y in range(numlag) for x in LFloc]
\fs24 \cb1 \

\fs28 \cb3 \'a0\'a0\'a0\'a0return featvec, featnames
\fs24 \cb1 \
\pard\pardeftab720\sl380\partightenfactor0

\fs28 \cf2 \
\
\
\pard\pardeftab720\sl380\partightenfactor0
\cf2 \cb3 def makeLagPredTar(numlag,LFloc,NLFloc,data,Tarname):
\fs24 \cb1 \

\fs28 \cb3 \'a0\'a0\'a0\'a0unused, fns = makeFV(numlag,numlag,data,LFloc,NLFloc)
\fs24 \cb1 \

\fs28 \cb3 \'a0\'a0\'a0\'a0DFpredictors = pd.DataFrame(columns = fns)
\fs24 \cb1 \

\fs28 \cb3 \'a0\'a0\'a0\'a0for ii in range(numlag,data.shape[0]):
\fs24 \cb1 \

\fs28 \cb3 \'a0\'a0\'a0\'a0DFpredictors.loc[ii], fns = makeFV(numlag,ii,data,LFloc,NLFloc)
\fs24 \cb1 \

\fs28 \cb3 \'a0\'a0\'a0\'a0DFtargets = data[Tarname][numlag:]
\fs24 \cb1 \

\fs28 \cb3 \'a0\'a0\'a0\'a0return DFpredictors, pd.DataFrame(DFtargets)
\fs24 \cb1 \
\pard\pardeftab720\sl380\partightenfactor0

\fs28 \cf2 \
\
\
\pard\pardeftab720\sl380\partightenfactor0
\cf2 \cb3 def makeLagTarOnly(numlag,data,Tarname):
\fs24 \cb1 \

\fs28 \cb3 \'a0\'a0\'a0\'a0return pd.DataFrame(data[Tarname][numlag:])
\fs24 \cb1 \
\pard\pardeftab720\sl380\partightenfactor0

\fs28 \cf2 \
\
\
\pard\pardeftab720\sl380\partightenfactor0
\cf2 \cb3 def rank_features(X,Y,method='MI',continuous=True):
\fs24 \cb1 \

\fs28 \cb3 \'a0\'a0\'a0\'a0if method == 'MI':
\fs24 \cb1 \

\fs28 \cb3 \'a0\'a0\'a0\'a0\'a0\'a0\'a0\'a0if continuous:
\fs24 \cb1 \

\fs28 \cb3 \'a0\'a0\'a0\'a0\'a0\'a0\'a0\'a0\'a0\'a0\'a0\'a0MIh = MIfsmethr(X,Y)
\fs24 \cb1 \

\fs28 \cb3 \'a0\'a0\'a0\'a0\'a0\'a0\'a0\'a0else:
\fs24 \cb1 \

\fs28 \cb3 \'a0\'a0\'a0\'a0\'a0\'a0\'a0\'a0\'a0\'a0\'a0\'a0MIh = MIfsmeth(X,Y)
\fs24 \cb1 \

\fs28 \cb3 \'a0\'a0\'a0\'a0\'a0\'a0\'a0\'a0\'a0\'a0\'a0\'a0FI_orderh = list(np.argsort(MIh)[::-1])
\fs24 \cb1 \

\fs28 \cb3 \'a0\'a0\'a0\'a0if method == 'MI2':
\fs24 \cb1 \

\fs28 \cb3 \'a0\'a0\'a0\'a0\'a0\'a0\'a0\'a0MIh =[]
\fs24 \cb1 \

\fs28 \cb3 \'a0\'a0\'a0\'a0\'a0\'a0\'a0\'a0for xcol in range(X.shape[1]):
\fs24 \cb1 \

\fs28 \cb3 \'a0\'a0\'a0\'a0\'a0\'a0\'a0\'a0\'a0\'a0\'a0\'a0MIh.append(lmi.mutual_information((X.iloc[:, xcol].values.reshape(-1, 1), Y.values.reshape(-1, 1)),k=4))
\fs24 \cb1 \

\fs28 \cb3 \'a0\'a0\'a0\'a0\'a0\'a0\'a0\'a0\'a0\'a0\'a0\'a0FI_orderh = list(np.argsort(MIh)[::-1])
\fs24 \cb1 \

\fs28 \cb3 \'a0\'a0\'a0\'a0return FI_orderh
\fs24 \cb1 \
\pard\pardeftab720\sl380\partightenfactor0

\fs28 \cf2 \
\
\
\pard\pardeftab720\sl380\partightenfactor0
\cf2 \cb3 def normalizeColumn(datacp,colname,rnfact):
\fs24 \cb1 \

\fs28 \cb3 \'a0\'a0\'a0\'a0datacp[colname] = datacp[colname] / float(rnfact)
\fs24 \cb1 \

\fs28 \cb3 \'a0\'a0\'a0\'a0return datacp
\fs24 \cb1 \
}