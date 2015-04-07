from __future__ import print_function, division
import numpy as np
import pandas as pd
import string
import unicodedata

def reapportionSeats(popv,includeOtherParties=False,method='state',nSeats=435,printDebug=False):
    """Reads a popular vote dictionary and reapportions congressional seats
    using the method of equal proportions at the state level (by party).
    http://en.wikipedia.org/wiki/United_States_congressional_apportionment#The_Method_of_Equal_Proportions
    
    Can be reapportioned within a state (method='state') or across all states (method='country')
    Returns a popular vote dictionary with expected number of representatives per state.

    NB: when method='country', states can end up with more or fewer seats than they actually have, which might be a problem!
    
    When method='state', makes some simplifying assumptions (see code).
    Every state with more than 2 seats starts with one seat per party.
    This is why we're not really ready for three parties, because "other" parties shouldn't always get a seat.
    """
    if includeOtherParties:
        print('Not ready for three parties!')
        return popv

    if includeOtherParties:
        nParties = 3
    else:
        nParties = 2

    # initialize to hold the popular vote per party
    partyPops = np.zeros([nParties,])

    if method == 'state':
        for st in popv.keys():
            partyPops[0] = popv[st]['demVotes']
            partyPops[1] = popv[st]['repVotes']
            if includeOtherParties:
                partyPops[2] = popv[st]['othVotes']
            stateVotes = sum(partyPops)
            popv[st]['stateVotes'] = stateVotes

            # real seats
            realSeats = np.array([popv[st]['demDistr'], popv[st]['repDistr'], popv[st]['othDistr']])
            nSeats = sum(realSeats)

            if nSeats == 1:
                expectedSeats = np.zeros([nParties,], dtype=np.int)
                expectedSeats[np.argmax(partyPops)] += 1
            elif nSeats == 2:
                # simplifying assumption: if there are two seats, divide equally
                # unless one party's popular vote is 2x that of the other, then give both seats to that party
                popProp = [i / float(stateVotes) for i in partyPops]
                if popProp[0] >= popProp[1] * 2:
                    expectedSeats = np.array([2, 0])
                elif popProp[1] >= popProp[0] * 2:
                    expectedSeats = np.array([0, 2])
                else:
                    expectedSeats = np.ones([nParties,], dtype=np.int)
                if printDebug:
                    print(popv[st]['state'] + st)
                    print(popProp)
                    print('party populations: ')
                    print(partyPops)
                    print('real seats: ')
                    print(realSeats)
                    print('expected seats: ')
                    print(expectedSeats)
            else:
                # every state with more than 2 seats starts with one seat per party
                expectedSeats = np.ones([nParties,], dtype=np.int)
                # allocate seats via the equal proportions method
                for i in range(nSeats - nParties):
                    # divide the party populations by the current value of the seats being doled out
                    currentProportion = np.array([x/y for x, y in zip(partyPops, expectedSeats)])
                    # since we're starting with 1, this isinf line isn't necessary
                    #currentProportion[np.isinf(currentProportion)] = 0
                    # add one to every party's seats to figure out who is most deserving based on populations and current seats
                    possSeats = [expectedSeats[j]+1 for j in range(len(expectedSeats))]
                    mostDeserving = np.argmax(np.multiply(currentProportion, [x/y for x, y in zip(partyPops,possSeats)]))
                    expectedSeats[mostDeserving] += 1
            popv[st]['demDistExpect'] = np.int(expectedSeats[0])
            popv[st]['repDistExpect'] = np.int(expectedSeats[1])
            if includeOtherParties:
                popv[st]['othDistExpect'] = np.int(expectedSeats[2])
            else:
                popv[st]['othDistExpect'] = 0
    elif method == 'country':
        for st in popv.keys():
            #statePop = float(popv[st]['demVotes'] + popv[st]['repVotes'] + popv[st]['othVotes'])
            #nSeats = popv[st]['demDistr'] + popv[st]['repDistr'] + popv[st]['othDistr']
            partyPops[0] += popv[st]['demVotes']
            partyPops[1] += popv[st]['repVotes']
            if includeOtherParties:
                partyPops[2] += popv[st]['othVotes']

        if printDebug:
            print('total party populations across the country')
            print(partyPops)

        totalPop = np.sum(partyPops)

        seatProp = [(x / float(totalPop)) * nSeats for x in partyPops]
        if printDebug:
            print('seat proportions')
            print(seatProp)
            print('rounded seat proportions')
            print(np.rint(seatProp))

        # rounding democratic and republican seats up, other party seats down.
        # this is essentially hardcoded
        if includeOtherParties:
            nDemSeats = int(np.ceil(seatProp[0]))
            nRepSeats = int(np.ceil(seatProp[1]))
            nOthSeats = int(np.floor(seatProp[2]))
        else:
            nDemSeats = int(np.round(seatProp[0]))
            nRepSeats = int(np.round(seatProp[1]))
            nOthSeats = 0

        if nDemSeats + nRepSeats + nOthSeats != nSeats:
            print('Rounding is not adding up to %d seats' % nSeats)
            return popv

        nStates = len(popv)

        # delegation size
        demSeats = np.ones(nStates, dtype=np.float)
        repSeats = np.ones(nStates, dtype=np.float)
        othSeats = np.zeros(nStates, dtype=np.float)

        # collect the number of votes per state per party
        demVotes = np.array([], dtype=np.float)
        repVotes = np.array([], dtype=np.float)
        if includeOtherParties:
            othVotes = np.array([], dtype=np.float)
        allStates = []
        for st in popv.keys():
            demVotes = np.append(demVotes, np.float(popv[st]['demVotes']))
            repVotes = np.append(repVotes, np.float(popv[st]['repVotes']))
            if includeOtherParties:
                othVotes = np.append(othVotes, np.float(popv[st]['othVotes']))
            allStates.append(popv[st]['state'])

        # allocate seats via the equal proportions method
        for i in range(nDemSeats - nStates):
            possSeats = [demSeats[j]+1 for j in range(len(demSeats))]
            mostDeserving = np.argmax(np.multiply([x/y for x, y in zip(demVotes, demSeats)], [x/y for x, y in zip(demVotes,possSeats)]))
            demSeats[mostDeserving] += 1

        for i in range(nRepSeats - nStates):
            possSeats = [repSeats[j]+1 for j in range(len(repSeats))]
            mostDeserving = np.argmax(np.multiply([x/y for x, y in zip(repVotes, repSeats)], [x/y for x, y in zip(repVotes,possSeats)]))
            repSeats[mostDeserving] += 1

        if includeOtherParties:
            # give out othSeats to the top states with the most oth votes (but we should divvy out based on proportion?)
            othStates = allStates[:]
            othVotesSorted = np.argsort(othVotes)
            othVotesSorted = othVotesSorted[::-1]
            othStatesSorted = [othStates[i] for i in othVotesSorted]
            for i in range(nOthSeats):
                othSeats[othVotesSorted[:nOthSeats][i]] += 1
        for st in popv.keys():
            stateIdx = allStates.index(popv[st]['state'])
            popv[st]['demDistExpect'] = np.int(demSeats[stateIdx])
            popv[st]['repDistExpect'] = np.int(repSeats[stateIdx])
            popv[st]['othDistExpect'] = np.int(othSeats[stateIdx])
    else:
        print('method must be \'state\' or \'country\'')
    return popv

def create_df_dist(popv,comp,dist,dfcensus,dfstatepvi,pviDict):
    """Give this method data for:
    popular vote (popv),
    compactness (comp),
    redistricting information like method and control (dist),
    census data (dfcensus),
    state data with PVI (dfstatepvi),
    and district-level PVI (pviDict)

    and output a dataframe consolidating this information.
    """
    # could also include governor's party, and whether redistricting is vetoproof (it is not in NH and RI); from dist dictionary

    state = []
    fips = []
    cd = []
    ficd = []

    pvi2010_state = []
    pvi2014_state = []

    lastName = []
    firstName = []
    party = []
    compact = []
    pvi_dist = []

    nVotes_dist = []
    nDemVotes_dist = []
    nRepVotes_dist = []
    nOthVotes_dist = []

    nDist = []
    nDemDist = []
    nRepDist = []
    nOthDist = []

    nDemDistExpect = []
    nRepDistExpect = []
    nOthDistExpect = []

    # delta_D_act_expect = []
    delta_R_act_expect = []
    # delta_O_act_expect = []

    # delta_D_act_expect_perc = []
    delta_R_act_expect_perc = []
    # delta_O_act_expect_perc = []

    redist_method = [] # Bipartisan, OneDistrict, or Nonpartisan (Nebraska only)
    redist_ctrl = [] # stored in redistricting_2010.json: R, D, Split, OneDistrict, or Nonpartisan (Nebraska only)

    census_types = ['white', 'black', 'hispa', 'asian', 'nativ', 'other', 'total']
    # initialize dict to store census data (percentage of state)
    census = dict((el,[]) for el in census_types)

    # for name processing
    punctTable = string.maketrans("","")

    for idx, st in enumerate(popv.keys()):
        nDistricts = sum([popv[st]['demDistr'], popv[st]['repDistr'], popv[st]['othDistr']])
        
        for d in range(1,nDistricts+1):
            if nDistricts == 1:
                cd_str = '00'
                stcd_str = '%s-AL' % popv[st]['state']
            else:
                cd_str = str(d)
                if len(cd_str) == 1:
                    cd_str = '0' + cd_str
                stcd_str = '%s-%s' % (popv[st]['state'], cd_str)
            
            ficd_str = st + cd_str
            state.append(popv[st]['state'])
            fips.append(st)
            cd.append(cd_str)
            ficd.append(ficd_str)

            compact.append(comp[ficd_str])

            for ct in census_types:
                census[ct].append(int(dfcensus[ct][dfcensus['state'] == stcd_str]))

            # redistricting method
            redist_method.append(dist[st]['redist_committee'])
            # redistricting control
            redist_ctrl.append(dist[st]['redist_ctrl'])

            # winner info; turn everything to ascii text, get first and last name, removing any punctuation
            lName, fName = splitName(pviDict[ficd_str]['name'][:-4], punctTable)
            lastName.append(lName)
            firstName.append(fName)

            thisParty = pviDict[ficd_str]['name'][-2].upper()
            if thisParty not in ['D', 'R', 'I']:
                print('%s is not a party, appending anyway')
            party.append(thisParty)

            # district PVI 114th congress
            pvi_dist.append(pviDict[ficd_str]['score'])
            
            # State PVI 2010
            score2010 = dfstatepvi.pvi2010[dfstatepvi.state == popv[st]['state']].get_values()
            pvi_party = score2010[0][0]
            if pvi_party == 'E':
                score2010 = 0
            elif pvi_party == 'D':
                score2010 = int(score2010[0][2:])
            elif pvi_party == 'R':
                score2010 = -int(score2010[0][2:])
            popv[st]['pvi2010'] = score2010
            pvi2010_state.append(score2010)

            # State PVI 2014
            score2014 = dfstatepvi.pvi2014[dfstatepvi.state == popv[st]['state']].get_values()
            pvi_party = score2014[0][0]
            if pvi_party == 'E':
                score2014 = 0
            elif pvi_party == 'D':
                score2014 = int(score2014[0][2:])
            elif pvi_party == 'R':
                score2014 = -int(score2014[0][2:])
            popv[st]['pvi2014'] = score2014
            pvi2014_state.append(score2014)

            nVotes_dist.append(sum([popv[st][cd_str]['demVotes'], popv[st][cd_str]['repVotes'], popv[st][cd_str]['othVotes']]))
            nDemVotes_dist.append(popv[st][cd_str]['demVotes'])
            nRepVotes_dist.append(popv[st][cd_str]['repVotes'])
            nOthVotes_dist.append(popv[st][cd_str]['othVotes'])

            nDist.append(nDistricts)
            nDemDist.append(popv[st]['demDistr'])
            nRepDist.append(popv[st]['repDistr'])
            nOthDist.append(popv[st]['othDistr'])

            nDemDistExpect.append(popv[st]['demDistExpect'])
            nRepDistExpect.append(popv[st]['repDistExpect'])
            nOthDistExpect.append(popv[st]['othDistExpect'])
            
            # straight subtraction between actual and expected number of seats
            #delta_D_act_expect.append(popv[st]['demDistr'] - popv[st]['demDistExpect'])
            delta_R_act_expect.append(popv[st]['repDistr'] - popv[st]['repDistExpect'])
            #delta_O_act_expect.append(popv[st]['othDistr'] - popv[st]['othDistExpect'])

            # subtraction (actual - expected number of seats) as percentage by number of districts
            #delta_D_act_expect_perc.append(((popv[st]['demDistr'] - popv[st]['demDistExpect']) / float(nDistricts)) * 100)
            delta_R_act_expect_perc.append(((popv[st]['repDistr'] - popv[st]['repDistExpect']) / float(nDistricts)) * 100)
            #delta_O_act_expect_perc.append(((popv[st]['othDistr'] - popv[st]['othDistExpect']) / float(nDistricts)) * 100)

            # percentage subtraction between actual and expected number of seats
            #delta_D_act_expect_perc.append((popv[st]['demDistr'] / float(nDistricts) * 100) - (popv[st]['demDistExpect'] / float(nDistricts) * 100))
            #delta_R_act_expect_perc.append((popv[st]['repDistr'] / float(nDistricts) * 100) - (popv[st]['repDistExpect'] / float(nDistricts) * 100))
            #delta_O_act_expect_perc.append((popv[st]['othDistr'] / float(nDistricts) * 100) - (popv[st]['othDistExpect'] / float(nDistricts) * 100))
            
    # create the dataframe

    # columns = ['compact', 'delta_R', 'delta_R', 'delta_O']
    # columns = ['state', 'fips', 'compact', 'pvi2010', 'pvi2014', 'delta_R', 'redist_method', 'redist_ctrl']

    # df = pd.DataFrame(index=statenum, columns=columns)
    df = pd.DataFrame()

    # df['party']=govparty
    # df['statenum'] = statenum
    df['state'] = state
    df['fips'] = fips
    df['cd'] = cd
    df['ficd'] = ficd

    df['pvi2010_state'] = pvi2010_state
    df['pvi2014_state'] = pvi2014_state

    df['lastName'] = lastName
    df['firstName'] = firstName
    df['party'] = party
    df['compact'] = compact
    df['pvi_dist'] = pvi_dist

    df['nVotes_dist'] = nVotes_dist
    df['nDemVotes_dist'] = nDemVotes_dist
    df['nRepVotes_dist'] = nRepVotes_dist
    df['nOthVotes_dist'] = nOthVotes_dist

    df['nDistricts'] = nDist
    df['nDemDist'] = nDemDist
    df['nRepDist'] = nRepDist
    df['nOthDist'] = nOthDist

    df['nDemDistExpect'] = nDemDistExpect
    df['nRepDistExpect'] = nRepDistExpect
    df['nOthDistExpect'] = nOthDistExpect

    df['delta_R'] = delta_R_act_expect
    df['delta_R_perc'] = delta_R_act_expect_perc

    df['redist_method'] = redist_method
    df['redist_ctrl'] = redist_ctrl

    # df['cens_total'] = census['total']
    df['cens_white'] = census['white']
    df['cens_black'] = census['black']
    df['cens_hispa'] = census['hispa']
    df['cens_asian'] = census['asian']
    df['cens_nativ'] = census['nativ']
    df['cens_other'] = census['other']

    return df

def summary_dist2state(df):
    """Receive a district dataframe made by create_df_dist() and summarize at the state level.
    """
    # data to get from the district summary; will store this exact data for a given state
    data_to_get_once = ['state', 'fips', 'pvi2010_state', 'pvi2014_state', 'nDistricts', 'nDemDist', 'nRepDist', 'nOthDist', 'nDemDistExpect', 'nRepDistExpect', 'nOthDistExpect', 'delta_R', 'delta_R_perc', 'redist_method', 'redist_ctrl']

    # data from our gerrymandering score calculation
    if 'gerryScore' in df.columns:
        data_to_get_once.append('gerryScore')
    if 'gerryWeight' in df.columns:
        data_to_get_once.append('gerryWeight')
    if 'ctrlPartySeatAdvantage' in df.columns:
        data_to_get_once.append('ctrlPartySeatAdvantage')

    # data to get from the district summary and aggregate over (by summing);
    # this is how columns are named in the district summary df
    data_to_sum_orig = ['nVotes_dist', 'nDemVotes_dist', 'nRepVotes_dist', 'nOthVotes_dist', 'cens_white', 'cens_black', 'cens_hispa', 'cens_asian', 'cens_nativ', 'cens_other']
    # this is how columns should be named in the state summary
    data_to_sum_save = ['nVotes', 'nDemVotes', 'nRepVotes', 'nOthVotes', 'cens_white', 'cens_black', 'cens_hispa', 'cens_asian', 'cens_nativ', 'cens_other']

    data_once = dict((el,[]) for el in data_to_get_once)
    data_sum = dict((el,[]) for el in data_to_sum_save)

    compact = []
    compact_25 = []
    compact_75 = []
    compact_min = []
    compact_mean = []

    for idx, st in enumerate(df.state.unique()):
        thisState = df[df.state == st]
        
        compact.append(list(thisState.compact))
        compact_25.append(float(np.percentile(thisState.compact,[25])))
        compact_75.append(float(np.percentile(thisState.compact,[75])))
        compact_min.append(float(np.min(thisState.compact)))
        compact_mean.append(float(np.mean(thisState.compact)))

        for d in data_to_get_once:
            data_once[d].append(thisState[d].iloc[0])
        
        for j, d in enumerate(data_to_sum_orig):
            data_sum[data_to_sum_save[j]].append(thisState[d].sum())

    # make a state summary dataframe
    dfstate = pd.DataFrame()

    for d in data_to_get_once:
        dfstate[d] = data_once[d]
    for d in data_to_sum_save:
        dfstate[d] = data_sum[d]
    
    dfstate['compact'] = compact
    dfstate['compact_25'] = compact_25
    dfstate['compact_75'] = compact_75
    dfstate['compact_min'] = compact_min
    dfstate['compact_mean'] = compact_mean
    return dfstate

def splitName(thisName, punctTable=string.maketrans("","")):
    thisName = unicodedata.normalize("NFKD", thisName.decode("utf8").strip()) #D is for d-ecomposed
    thisName = u"".join([c for c in thisName if not unicodedata.combining(c)])
    thisName = thisName.encode('utf8').split(',', 1)
    # get first and last name, removing any punctuation and whitespace
    #lName = thisName[0].translate(punctTable, string.punctuation)
    #fName = thisName[1].translate(punctTable, string.punctuation).strip()
    #lName = "".join(lName.split())
    lName = "".join(thisName[0].translate(punctTable, string.punctuation).split()).title()
    fName = thisName[1].translate(punctTable, string.punctuation).strip().title()
    return lName, fName

def gerry_score(dfdist,minDist=4,packMin=70,packMax=99,crackMin=35,crackMax=49,scoreLim=0,delta_R_lim=0,printDebug=False):
    """calculate a gerrymandering score using the number of packed and cracked districts

    input:
        minDist: the minimum number of districts a state needs to be included

        packMin/packMax: the limits on vote proportion that one party needs in a given district to be packed

        crackMin/crackMax: the limits on vote proportion that one party needs to be cracked

        scoreLim: abs value.
                  proceed with gerrymandering assessment if packed+cracked comparison score is above/below this limit

        delta_R_lim: if scoreLim is surpassed, consider a state gerrymandered if delta seats larger than this number

        printDebug: print out information for gerrymandered states

    returns:
        gerryScore: the difference between packed and cracked seats for one party vs the other.
                    negative = Democratic advantage, positive = Republican advantage.

        gerryWeight: 1.0 if gerrymandered, 0.5 if gerrymandered but redistricting control is split, 0.0 otherwise

        ctrlPartySeatAdvantage: boolean.
                                whether the party in control of redistricting has the congressional seat advantage.
                                False for Split/Bipartisan/independent committee control.
    """

    # use unique so it preserves state order
    states = dfdist.state.unique()
    state_gerry = []
    state_nogerry = []
    state_noeval = []

    # convert votes to percentages; only use democratic and republican votes
    # totalVotes = df.nDemVotes_dist + df.nRepVotes_dist + df.nOthVotes_dist
    totalVotes = dfdist.nDemVotes_dist + dfdist.nRepVotes_dist
    dfdist['demVotes_perc'] = (dfdist.nDemVotes_dist / totalVotes) * 100
    dfdist['repVotes_perc'] = (dfdist.nRepVotes_dist / totalVotes) * 100
    # dfdist['othVotes_perc'] = (dfdist.nOthVotes_dist / totalVotes) * 100

    gerryScore = []
    gerryWeight = []
    ctrlPartySeatAdvantage = []
    seatDiff = 0

    for st in states:
        thisState = dfdist[dfdist['state']==st]
        nDist = thisState.nDistricts.iloc[0]
        
        # initialize
        thisScore = 0
        thisGerryWeight = 0.0
        ctrlPAdvantage = False
        
        if nDist >= minDist:
            packD = (thisState.demVotes_perc >= packMin) & (thisState.demVotes_perc <= packMax)
            crackD = (thisState.demVotes_perc >= crackMin) & (thisState.demVotes_perc <= crackMax)
            nPackD = sum(packD)
            nCrackD = sum(crackD)

            packR = (thisState.repVotes_perc >= packMin) & (thisState.repVotes_perc <= packMax)
            crackR = (thisState.repVotes_perc >= crackMin) & (thisState.repVotes_perc <= crackMax)
            nPackR = sum(packR)
            nCrackR = sum(crackR)

            thisScore = (nPackD + nCrackD - nPackR - nCrackR)
            packedD = thisState[packD]['cd'].values
            crackedD = thisState[crackD]['cd'].values
            packedR = thisState[packR]['cd'].values
            crackedR = thisState[crackR]['cd'].values
            
            delta_R = thisState.delta_R.iloc[0]
            
            redist_method = thisState.redist_method.iloc[0]
            redist_ctrl = thisState.redist_ctrl.iloc[0]
            
            nDem = thisState.nDemDist.iloc[0]
            nRep = thisState.nRepDist.iloc[0]
            if nDem > nRep:
                majParty = 'D majority'
            elif nDem < nRep:
                majParty = 'R majority'
            else:
                majParty = 'Split'
            
            gerryText = ''
            if (thisScore > scoreLim) and (delta_R > delta_R_lim):
                state_gerry.append(st)
                gerryText = 'republicans are gerrymandering!'
                thisGerryWeight = 1.0
                seatDiff += delta_R
                if majParty[0] == 'R' and redist_ctrl[0] == 'R':
                    ctrlPAdvantage = True
            elif (thisScore < -scoreLim) and (delta_R < -delta_R_lim):
                state_gerry.append(st)
                gerryText = 'democrats are gerrymandering!'
                thisGerryWeight = 1.0
                seatDiff += delta_R
                if majParty[0] == 'D' and redist_ctrl[0] == 'D':
                    ctrlPAdvantage = True
            else:
                state_nogerry.append(st)

            if gerryText != '' and redist_ctrl[0] == 'S':
                thisGerryWeight = 0.5
                gerryText += ' (probably not because redist control is Split, giving the seats back)'
                # give the seats back
                seatDiff -= delta_R

            if printDebug and len(gerryText) > 0:
                print('%s (%d dist), %s seats (D: %d, R: %d)'% (st, nDist, majParty, nDem, nRep))
                print(gerryText)
                print('\tnPackR: %d, nCrackR: %d' % (nPackR, nCrackR))
                print('\tnPackD: %d, nCrackD: %d' % (nPackD, nCrackD))
                print('\tpackedR = %s' % packedR)
                print('\tcrackedR = %s' % crackedR)
                print('\tpackedD = %s' % packedD)
                print('\tcrackedD = %s' % crackedD)
                print('\tgm score: %d, deltaSeats: %d' % (thisScore, thisState.delta_R.iloc[0]))
                print('\tredist ctrl: %s: %s; seat advantage: %s\n' % (redist_ctrl, redist_method, ctrlPAdvantage))
        else:
            state_noeval.append(st)

        gerryScore.extend(np.tile(thisScore,nDist))
        gerryWeight.extend(np.tile(thisGerryWeight,nDist))
        ctrlPartySeatAdvantage.extend(np.tile(ctrlPAdvantage,nDist))

    #if printDebug:
    print('seat difference: %d (positive indicates number of R seats that should be D)' % seatDiff)
    print(state_gerry)

    dfdist['gerryScore'] = gerryScore
    dfdist['gerryWeight'] = gerryWeight
    dfdist['ctrlPartySeatAdvantage'] = ctrlPartySeatAdvantage

    return (dfdist, state_gerry)
