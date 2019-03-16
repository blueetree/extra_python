def balanceData(train_x_h, train_y_h, cntar_h):
    L1 = train_x_h[cntar_h + "_lag_1"].tolist()
    L2 = train_y_h.tolist()
    nochangebool = [L1[xx] == L2[xx] for xx in range(len(L2))]
    changeinds = [xx for xx in range(len(nochangebool)) if not nochangebool[xx]]
    nochangeinds = [xx for xx in range(len(nochangebool)) if nochangebool[xx]]
    nochangess = np.random.choice(nochangeinds, len(changeinds), replace=False).tolist()
    nki = nochangess + changeinds
    train_x_f = train_x_h.iloc[nki]
    train_y_f = train_y_h.iloc[nki]
    return train_x_f, train_y_f, nki


def FilterFeatures(train_x_h,train_y_h,test_x_h,FSparams_h):
    if FSparams_h['do_FS']:
        FI_order = rank_features(train_x_h, train_y_h, method=FSparams_h['FSmethod'], continuous=True)
        SelectedFeatures_loc = FI_order[:FSparams_h['NSF']]
        train_x_f = train_x_h.iloc[:,SelectedFeatures_loc]
        test_x_f = test_x_h.iloc[:,SelectedFeatures_loc]
    else:
        train_x_f = train_x_h
        test_x_f = test_x_h
        SelectedFeatures_loc = -1
    return train_x_f, test_x_f, SelectedFeatures_loc



def fsloopsfromdict(FSparamdict_h):
    FSparamsLooplist_h = []
    if False in FSparamdict_h['do_FS']:
        FSparamsLooplist_h = FSparamsLooplist_h + [{'do_FS': False,'FSmethod': 'NoFS','NSF': 0}]
    if True in FSparamdict_h['do_FS']:
        IL =list(itertools.product([True],FSparamdict_h['FSmethod'],FSparamdict_h['NSF']))
        FSparamsLooplist_h = FSparamsLooplist_h + [{'do_FS': xx[0],'FSmethod': xx[1],'NSF': xx[2]} for xx in IL]
    return FSparamsLooplist_h



def AssemblePreds(FullPredsDict_l):
    FullPredsDict_l['FullPredictionsAssembled']={}
    foldind = FullPredsDict_l['FullPredictionsKeyMeaning'].index('Fold')
    fpk = [make_tuple(xx) for xx in FullPredsDict_l['FullPredictions'].keys()]
    fpkl = [list(xx) for xx in fpk]
    fpklm = [xx[:foldind] + xx[foldind + 1:] for xx in fpkl]
    fpklmu = [list(i) for i in set(tuple(i) for i in fpklm)]
    nwof = FullPredsDict_l['FullPredictionsKeyMeaning'][:foldind] + \
    FullPredsDict_l['FullPredictionsKeyMeaning'][foldind +1:]
    FullPredsDict_l['FullPredictionsAssembledKeyMeaning'] = nwof
    newrepind = nwof.index('Rep')
    for ml in fpklmu:
        FullPredsDict_l['FullPredictionsAssembled'][tuple(ml)]=np.full(len(FullPredsDict_l['Target']),np.nan)
        for fn in range(FullPredsDict_l['K']):
            predinds = FullPredsDict_l['RepFoldIndices'][ml[newrepind]][fn]
            fullkey = ml[:]
            fullkey.insert(foldind,fn)
            FullPredsDict_l['FullPredictionsAssembled'][tuple(ml)][predinds] = FullPredsDict_l['FullPredictions'][str(tuple(fullkey))]
    return FullPredsDict_l

def CalcMSEs(FullPredsDict_l):
    FullPredsDict_l['MSE']={}
    for thiskey in FullPredsDict_l['FullPredictionsAssembled'].keys():
        preds = FullPredsDict_l['FullPredictionsAssembled'][thiskey]
        FullPredsDict_l['MSE'][thiskey] = mean_squared_error(preds, FullPredsDict_l['Target']) # this was Tar..
    return FullPredsDict_l



def CalcMSE_mean_std(FullPredsDict_l):
    msekeys = FullPredsDict_l['MSE'].keys()
    repind = FullPredsDict_l['FullPredictionsAssembledKeyMeaning'].index("Rep")
    fpkl = [list(xx) for xx in msekeys]
    fpklm = [xx[:repind] + xx[repind + 1:] for xx in fpkl]
    fpklmu = [list(i) for i in set(tuple(i) for i in fpklm)]
    FullPredsDict_l['MSE_mean']={}
    FullPredsDict_l['MSE_std']={}
    for thiskey in fpklmu:
        tkc = thiskey[:]
        thesemses = []
        for rep in range(FullPredsDict_l["Reps"]):
            trk = tkc[:]
            trk.insert(repind,rep)
            thesemses.append(FullPredsDict_l['MSE'][tuple(trk)])
        FullPredsDict_l['MSE_mean'][tuple(thiskey)] = np.mean(thesemses)
        FullPredsDict_l['MSE_std'][tuple(thiskey)] = np.std(thesemses)
        FullPredsDict_l['MSE_mean_std_KeyMeaning'] =[xx for xx in FullPredsDict_l['FullPredictionsAssembledKeyMeaning'] if xx != "Rep"]
    return FullPredsDict_l



def mergeFullPreds(fpd1_in,fpd2):
    fpd1 = copy.deepcopy(fpd1_in)
    fpd1['SelectedFeatures'].update(fpd2['SelectedFeatures'])
    fpd1['FullPredictions'].update(fpd2['FullPredictions'])
    return fpd1


def PredictionFeedback(mod,dat,Target_loc,numlags):
    adjustcols = [Target_loc.split("_")[0] + "_lag_" + str(xx) for xx in range(1, numlags)]
    PredictionsList = []
    ds0 = dat.shape[0]
    bufferd = deque(dat.iloc[[0]][adjustcols].values[0])
    for sampnum in range(ds0):
        x1 = dat.iloc[[sampnum]].copy()
         x1.loc[x1.index[0], adjustcols] = bufferd
        Pred1 = mod.predict(x1)
        bufferd.appendleft(Pred1[0])
        bufferd.pop()
        PredictionsList.append(Pred1[0])
    return PredictionsList


def CheckTimestampOrder(dat):
    tsdt = pd.to_datetime(dat['timestamp'], format='%Y-%m-%d %H:%M:%S')
    return tsdt.tolist() == sorted(tsdt.tolist())


# take a look at the two sets of predictions vs true
def FullPlot(winpreds_h,Tar_h,ptss_h):
    fig,axarr = plt.subplots(1,sharex=True,figsize=(10, 1))
    axarr.plot(winpreds_h[0:ptss_h], color='blue')
    axarr.plot(Tar_h[0:ptss_h], color='red')
    box = axarr.get_position()
    axarr.set_position([box.x0, box.y0 +0.1, box.width * 0.8, box.height * 0.7])
    axarr.legend(['Predicted', 'True Value'], loc='center left', bbox_to_anchor=(1, 0.5))
    return plt



def FullPlot_ax(winpreds_h,Tar_h,ptss_h,axarr, ttl=""):
    # fig,axarr = plt.subplots(1,sharex=True,figsize=(10, 1))
    axarr.plot(winpreds_h[0:ptss_h], color='blue')
    axarr.plot(Tar_h[0:ptss_h], color='red')
    box = axarr.get_position()
    axarr.set_position([box.x0, box.y0 +0.1, box.width * 0.8, box.height * 0.7])
    axarr.legend(['Predicted', 'True Value'], loc='center left', bbox_to_anchor=(1, 0.5))
    axarr.set_title(ttl)
    return plt



def FeaturePlot_ax(TripData_h,ptss_h,axarr,ttl="Environment"):
    colset = cm.rainbow(np.linspace(0, 1, 8))
    axarr.plot((TripData_h['OtsAirTmpCrVal_lag_1'][0:ptss_h]/max(TripData_h['OtsAirTmpCrVal_lag_1'])).tolist(), color = colset[0])
    axarr.plot((TripData_h['IPSnsrSolrInt_lag_1'][0:ptss_h]/max(TripData_h['IPSnsrSolrInt_lag_1'])).tolist(), color = colset[1])
    axarr.plot((TripData_h['EngSpd_lag_1'][0:ptss_h]/max(TripData_h['EngSpd_lag_1'])).tolist(), color = colset[2])
    axarr.plot((TripData_h['DriverSetTemp_lag_1'][0:ptss_h]/max(TripData_h['DriverSetTemp_lag_1'])).tolist(), color = colset[3])
    axarr.plot((TripData_h['WindPattern_lag_1'][0:ptss_h]).tolist(), color = colset[4])
    axarr.plot((TripData_h['WindLevel_lag_1'][0:ptss_h]).tolist(), color = colset[5])
    axarr.plot((TripData_h['LftLoDctTemp_lag_1'][0:ptss_h]/max(TripData_h['LftLoDctTemp_lag_1'])).tolist(), color = colset[6])
    axarr.plot((TripData_h['LftUpDctTemp_lag_1'][0:ptss_h]/max(TripData_h['LftUpDctTemp_lag_1'])).tolist(), color = colset[7])
    #axarr.plot(Tar_h[0:ptss_h], color='red')
    box = axarr.get_position()
    axarr.set_position([box.x0, box.y0 +0.1, box.width * 0.8, box.height * 0.7])
    axarr.legend(['OtsAirTmpCrVal_lag_1', 'IPSnsrSolrInt_lag_1', 'EngSpd_lag_1', 'DriverSetTemp_lag_1', 'WindPattern_lag_1', 'WindLevel_lag_1','LftLoDctTemp_lag_1','LftUpDctTemp_lag_1'],
    loc='center left',
    fontsize=4,
    bbox_to_anchor=(1, 0.5))
    axarr.set_title(ttl)
    return plt


def stratified_cross_validation_splits(yy, K):
    outinds = [[] for x in range(K)]
    if isinstance(yy,pd.core.series.Series):
        y = yy.tolist()
    else:
        if not isinstance(yy,list):
            raise ValueError('Target values must be either a list or a pandas series')
        y=yy[:]
    ysi = np.argsort(y).tolist()
    bi = range(0,len(y),K)
    if len(y) not in bi:
        bi = bi +[len(y)]
    bir =[range(bi[x],bi[x+1]) for x in range(len(bi)-1)]
    for ii in range(len(bir)):
        random.shuffle(bir[ii])
    for bs in range(len(bir)):
        for bss in range(len(bir[bs])):
            outinds[bss]=outinds[bss] + [ysi[bir[bs].pop()]]
    return outinds

def check_make_path(pn):
    if not os.path.exists(pn):
        os.makedirs(pn)



def getTrainTest(data,targets,testinds):
    traininds = [x for x in range(data.shape[0]) if x not in testinds]
    train_x = data.iloc[traininds].reset_index(drop=True)
    test_x = data.iloc[testinds].reset_index(drop=True)
    train_y = targets.iloc[traininds].reset_index(drop=True)
    test_y = targets.iloc[testinds].reset_index(drop=True)
    return train_x, train_y, test_x, test_y



# implement stratified cross validation splits
def sss(yy,K):
    outinds = [[] for x in range(K)]
    if isinstance(yy,pd.core.series.Series):
        y = yy.tolist()
    else:
        if not isinstance(yy,list):
            raise ValueError('Target values must be either a list or a pandas series')
        y=yy[:]
    ysi = np.argsort(y).tolist()
    bi = range(0,len(y),K)
    if len(y) not in bi:
        bi = bi +[len(y)]
    bir =[range(bi[x],bi[x+1]) for x in range(len(bi)-1)]
    for ii in range(len(bir)):
        random.shuffle(bir[ii])
    for bs in range(len(bir)):
        for bss in range(len(bir[bs])):
            outinds[bss]=outinds[bss] + [ysi[bir[bs].pop()]]
    return outinds



def makeFV(numlag, rid, Data,LFloc,NLFloc):
    if rid < numlag:
    raise ValueError('Row number cannot be less than the lag')
    # this is assuming the rows are in temporal order - ensure this earlier
    FV = Data[NLFloc].iloc[rid]
    LFm = Data[LFloc].iloc[(rid-numlag+1):(rid+1)][::-1]
    LFv = flatten(LFm.as_matrix().tolist())
    featvec = FV.tolist() + LFv
    featnames = list(FV.index) + [x+"_lag_" + str(y) for y in range(numlag) for x in LFloc]
    return featvec, featnames



def makeLagPredTar(numlag,LFloc,NLFloc,data,Tarname):
    unused, fns = makeFV(numlag,numlag,data,LFloc,NLFloc)
    DFpredictors = pd.DataFrame(columns = fns)
    for ii in range(numlag,data.shape[0]):
    DFpredictors.loc[ii], fns = makeFV(numlag,ii,data,LFloc,NLFloc)
    DFtargets = data[Tarname][numlag:]
    return DFpredictors, pd.DataFrame(DFtargets)



def makeLagTarOnly(numlag,data,Tarname):
    return pd.DataFrame(data[Tarname][numlag:])



def rank_features(X,Y,method='MI',continuous=True):
    if method == 'MI':
        if continuous:
            MIh = MIfsmethr(X,Y)
        else:
            MIh = MIfsmeth(X,Y)
            FI_orderh = list(np.argsort(MIh)[::-1])
    if method == 'MI2':
        MIh =[]
        for xcol in range(X.shape[1]):
            MIh.append(lmi.mutual_information((X.iloc[:, xcol].values.reshape(-1, 1), Y.values.reshape(-1, 1)),k=4))
            FI_orderh = list(np.argsort(MIh)[::-1])
    return FI_orderh



def normalizeColumn(datacp,colname,rnfact):
    datacp[colname] = datacp[colname] / float(rnfact)
    return datacp
