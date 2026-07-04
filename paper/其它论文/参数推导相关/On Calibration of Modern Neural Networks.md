|     |     |     |            | On Calibration |     |               | of Modern | Neural  |                     | Networks |     |              |     |     |
| --- | --- | --- | ---------- | -------------- | --- | ------------- | --------- | ------- | ------------------- | -------- | --- | ------------ | --- | --- |
|     |     |     | ChuanGuo*1 |                |     | GeoffPleiss*1 |           | YuSun*1 | KilianQ.Weinberger1 |          |     |              |     |     |
|     |     |     | Abstract   |                |     |               |           |         | LeNet(1998)         |          |     | ResNet(2016) |     |     |
|     |     |     |            |                |     |               |           |         | CIFAR-100           |          |     | CIFAR-100    |     |     |
1.0
Confidencecalibration–theproblemofpredict-
|     |                 |     |           |                |     |     |     |     |     | ecnedfinoc ycaruccA |     |     | ycaruccA | ecnedfinoc |
| --- | --------------- | --- | --------- | -------------- | --- | --- | --- | --- | --- | ------------------- | --- | --- | -------- | ---------- |
|     | ing probability |     | estimates | representative |     | of  | the |     |     |                     |     |     |          |            |
0.8
| 7102 guA 3  ]GL.sc[  2v99540.6071:viXra |                  |     |            |     |                |     |     | selpmaSfo% |     |     |     |     |     |     |
| --------------------------------------- | ---------------- | --- | ---------- | --- | -------------- | --- | --- | ---------- | --- | --- | --- | --- | --- | --- |
|                                         | true correctness |     | likelihood |     | – is important |     | for |            |     |     |     |     |     |     |
0.6
|     | classification                            |        | models      | in many    | applications. |             | We  |     |     |      |     |     |     |      |
| --- | ----------------------------------------- | ------ | ----------- | ---------- | ------------- | ----------- | --- | --- | --- | ---- | --- | --- | --- | ---- |
|     | discover                                  | that   | modern      | neural     | networks,     | unlike      |     |     |     |      |     |     |     |      |
|     |                                           |        |             |            |               |             |     | 0.4 |     | .gvA |     |     |     | .gvA |
|     | those from                                | a      | decade ago, | are        | poorly        | calibrated. |     |     |     |      |     |     |     |      |
|     | Throughextensiveexperiments,weobservethat |        |             |            |               |             |     | 0.2 |     |      |     |     |     |      |
|     | depth,                                    | width, | weight      | decay, and | Batch         | Normal-     |     | 0.0 |     |      |     |     |     |      |
izationareimportantfactorsinfluencingcalibra- 0.0 0.2 0.4 0.6 0.8 1.0 0.0 0.2 0.4 0.6 0.8 1.0
|     | tion.           | We evaluate   | the         | performance |     | of various   |     | 1.0 |     |         |     |         |     |     |
| --- | --------------- | ------------- | ----------- | ----------- | --- | ------------ | --- | --- | --- | ------- | --- | ------- | --- | --- |
|     |                 |               |             |             |     |              |     |     |     | Outputs |     | Outputs |     |     |
|     | post-processing |               | calibration | methods     |     | on state-of- |     | 0.8 |     |         |     |         |     |     |
|     |                 |               |             |             |     |              |     |     |     | Gap     |     | Gap     |     |     |
|     | the-art         | architectures | with        | image       | and | document     |     |     |     |         |     |         |     |     |
ycaruccA
|     | classification |          | datasets. | Our analysis |      | and exper- |      | 0.6 |     |     |     |     |     |     |
| --- | -------------- | -------- | --------- | ------------ | ---- | ---------- | ---- | --- | --- | --- | --- | --- | --- | --- |
|     | iments         | not only | offer     | insights     | into | neural     | net- |     |     |     |     |     |     |     |
0.4
|     | work learning, |     | but also | provide | a   | simple | and |     |     |     |     |     |     |     |
| --- | -------------- | --- | -------- | ------- | --- | ------ | --- | --- | --- | --- | --- | --- | --- | --- |
0.2
|     | straightforward |     | recipe      | for practical | settings: |             | on  |     |     |            |     |     |            |     |
| --- | --------------- | --- | ----------- | ------------- | --------- | ----------- | --- | --- | --- | ---------- | --- | --- | ---------- | --- |
|     |                 |     |             |               |           |             |     |     |     | Error=44.9 |     |     | Error=30.6 |     |
|     | most datasets,  |     | temperature | scaling       |           | – a single- |     |     |     |            |     |     |            |     |
0.0
|     | parameter                               | variant | of  | Platt Scaling | –   | is surpris- |     |                    |     |            |            |                 |         |          |
| --- | --------------------------------------- | ------- | --- | ------------- | --- | ----------- | --- | ------------------ | --- | ---------- | ---------- | --------------- | ------- | -------- |
|     |                                         |         |     |               |     |             |     | 0.0                | 0.2 | 0.4 0.6    | 0.8 1.0    | 0.0 0.2         | 0.4 0.6 | 0.8 1.0  |
|     | inglyeffectiveatcalibratingpredictions. |         |     |               |     |             |     |                    |     |            | Confidence |                 |         |          |
|     |                                         |         |     |               |     |             |     | Figure1.Confidence |     | histograms | (top)      | and reliability |         | diagrams |
(bottom)fora5-layerLeNet(left)anda110-layerResNet(right)
1.Introduction
onCIFAR-100.Refertothetextbelowfordetailedillustration.
| Recent | advances |         | in deep learning |           | have dramatically |              | im- |                  |     |         |        |                     |     |         |
| ------ | -------- | ------- | ---------------- | --------- | ----------------- | ------------ | --- | ---------------- | --- | ------- | ------ | ------------------- | --- | ------- |
|        |          |         |                  |           |                   |              |     | If the detection |     | network | is not | able to confidently |     | predict |
| proved | neural   | network | accuracy         | (Simonyan |                   | & Zisserman, |     |                  |     |         |        |                     |     |         |
thepresenceorabsenceofimmediateobstructions,thecar
2015;Srivastavaetal.,2015;Heetal.,2016;Huangetal.,
shouldrelymoreontheoutputofothersensorsforbraking.
2016;2017).Asaresult,neuralnetworksarenowentrusted
|     |     |     |     |     |     |     |     | Alternatively, |     | in automated | health | care, control | should | be  |
| --- | --- | --- | --- | --- | --- | --- | --- | -------------- | --- | ------------ | ------ | ------------- | ------ | --- |
withmakingcomplexdecisionsinapplications,suchasob-
passedontohumandoctorswhentheconfidenceofadis-
| ject   | detection | (Girshick, | 2015),      | speech    | recognition |          | (Han-     |                                             |             |        |                 |              |              |         |
| ------ | --------- | ---------- | ----------- | --------- | ----------- | -------- | --------- | ------------------------------------------- | ----------- | ------ | --------------- | ------------ | ------------ | ------- |
|        |           |            |             |           |             |          |           | easediagnosisnetworkislow(Jiangetal.,2012). |             |        |                 |              |              | Specif- |
| nun    | et al.,   | 2014),     | and medical | diagnosis |             | (Caruana | et al.,   |                                             |             |        |                 |              |              |         |
|        |           |            |             |           |             |          |           | ically,                                     | a network   | should | provide         | a calibrated | confidence   |         |
| 2015). | In these  | settings,  | neural      | networks  |             | are an   | essential |                                             |             |        |                 |              |              |         |
|        |           |            |             |           |             |          |           | measure                                     | in addition | to     | its prediction. | In           | other words, | the     |
componentoflargerdecisionmakingpipelines.
probabilityassociatedwiththepredictedclasslabelshould
In real-world decision making systems, classification net- reflectitsgroundtruthcorrectnesslikelihood.
| works                           | must | not only | be accurate, |     | but also     | should | indicate |                        |            |                                |           |          |           |     |
| ------------------------------- | ---- | -------- | ------------ | --- | ------------ | ------ | -------- | ---------------------- | ---------- | ------------------------------ | --------- | -------- | --------- | --- |
|                                 |      |          |              |     |              |        |          | Calibrated             | confidence |                                | estimates | are also | important | for |
| whentheyarelikelytobeincorrect. |      |          |              |     | Asanexample, |        | con-     |                        |            |                                |           |          |           |     |
|                                 |      |          |              |     |              |        |          | modelinterpretability. |            | Humanshaveanaturalcognitivein- |           |          |           |     |
sideraself-drivingcarthatusesaneuralnetworktodetect
|             |     |           |              |     |           |         |        | tuitionforprobabilities(Cosmides&Tooby,1996). |           |         |     |                |     | Good      |
| ----------- | --- | --------- | ------------ | --- | --------- | ------- | ------ | --------------------------------------------- | --------- | ------- | --- | -------------- | --- | --------- |
| pedestrians |     | and other | obstructions |     | (Bojarski | et al., | 2016). |                                               |           |         |     |                |     |           |
|             |     |           |              |     |           |         |        | confidence                                    | estimates | provide | a   | valuable extra | bit | of infor- |
*Equal 1Cornell mation to establish trustworthiness with the user – espe-
|     | contribution, |     | alphabetical | order. |     | University. |     |     |     |     |     |     |     |     |
| --- | ------------- | --- | ------------ | ------ | --- | ----------- | --- | --- | --- | --- | --- | --- | --- | --- |
Correspondence to: Chuan Guo <cg563@cornell.edu>, Geoff cially for neural networks, whose classification decisions
Pleiss<geoff@cs.cornell.edu>,YuSun<ys646@cornell.edu>. are often difficult to interpret. Further, good probability
|     |     |     |     |     |     |     |     | estimates | can | be used to | incorporate | neural | networks | into |
| --- | --- | --- | --- | --- | --- | --- | --- | --------- | --- | ---------- | ----------- | ------ | -------- | ---- |
34th
| Proceedings |         | of the     | International |      | Conference | on        | Machine |                           |     |              |                          |         |      |        |
| ----------- | ------- | ---------- | ------------- | ---- | ---------- | --------- | ------- | ------------------------- | --- | ------------ | ------------------------ | ------- | ---- | ------ |
|             |         |            |               |      |            |           |         | otherprobabilisticmodels. |     |              | Forexample,onecanimprove |         |      |        |
| Learning,   | Sydney, | Australia, |               | PMLR | 70, 2017.  | Copyright | 2017    |                           |     |              |                          |         |      |        |
|             |         |            |               |      |            |           |         | performance               |     | by combining | network                  | outputs | with | a lan- |
bytheauthor(s).

guage model in speech recognition (Hannun et al., 2014; 0.8,weexpectthat80shouldbecorrectlyclassified. More
Xiongetal.,2016),orwithcamerainformationforobject formally,wedefineperfectcalibrationas
| detection(Kendall&Cipolla,2016). |        |                |           |                 |        |             |           |               | (cid:16) Yˆ | |Pˆ       | (cid:17)    |             |               |           |
| -------------------------------- | ------ | -------------- | --------- | --------------- | ------ | ----------- | --------- | ------------- | ----------- | --------- | ----------- | ----------- | ------------- | --------- |
|                                  |        |                |           |                 |        |             |           | P             | =Y          |           | =p =p,      |             | ∀p∈[0,1]      | (1)       |
| In 2005, Niculescu-Mizil         |        |                | & Caruana |                 | (2005) | showed that |           |               |             |           |             |             |               |           |
|                                  |        |                |           |                 |        |             | where     | the           | probability | is        | over the    | joint       | distribution. | In all    |
| neural networks                  |        | typically      | produce   | well-calibrated |        | proba-      |           |               |             |           |             |             |               |           |
|                                  |        |                |           |                 |        |             | practical | settings,     |             | achieving | perfect     | calibration |               | is impos- |
| bilities on                      | binary | classification |           | tasks.          | While  | neural net- |           |               |             |           |             |             |               |           |
|                                  |        |                |           |                 |        |             | sible.    | Additionally, |             | the       | probability | in          | (1) cannot    | be com-   |
workstodayareundoubtedlymoreaccuratethantheywere
putedusingfinitelymanysamplessincePˆ
isacontinuous
| a decade ago, | we       | discover | with      | great            | surprise | that mod- |                 |     |                                     |     |     |     |     |     |
| ------------- | -------- | -------- | --------- | ---------------- | -------- | --------- | --------------- | --- | ----------------------------------- | --- | --- | --- | --- | --- |
|               |          |          |           |                  |          |           | randomvariable. |     | Thismotivatestheneedforempiricalap- |     |     |     |     |     |
| ern neural    | networks | are      | no longer | well-calibrated. |          | This      |                 |     |                                     |     |     |     |     |     |
proximationsthatcapturetheessenceof(1).
isvisualizedinFigure1,whichcomparesa5-layerLeNet
(left)(LeCunetal.,1998)witha110-layerResNet(right)
(He et al., 2016) on the CIFAR-100 dataset. The top row ReliabilityDiagrams (e.g. Figure1bottom)areavisual
|                                                 |     |     |     |     |     |       | representation |     | of  | model | calibration | (DeGroot |     | & Fienberg, |
| ----------------------------------------------- | --- | --- | --- | --- | --- | ----- | -------------- | --- | --- | ----- | ----------- | -------- | --- | ----------- |
| showsthedistributionofpredictionconfidence(i.e. |     |     |     |     |     | prob- |                |     |     |       |             |          |     |             |
abilitiesassociatedwiththepredictedlabel)ashistograms. 1983;Niculescu-Mizil&Caruana,2005). Thesediagrams
plotexpectedsampleaccuracyasafunctionofconfidence.
TheaverageconfidenceofLeNetcloselymatchesitsaccu-
racy,whiletheaverageconfidenceoftheResNetissubstan- Ifthemodelisperfectlycalibrated–i.e. if(1)holds–then
tiallyhigherthanitsaccuracy. Thisisfurtherillustratedin the diagram should plot the identity function. Any devia-
tionfromaperfectdiagonalrepresentsmiscalibration.
thebottomrowreliabilitydiagrams(DeGroot&Fienberg,
1983;Niculescu-Mizil&Caruana,2005),whichshowac-
Toestimatetheexpectedaccuracyfromfinitesamples,we
| curacy as | a function | of  | confidence. | We  | see | that LeNet is |                       |     |     |     |                             |     |     |     |
| --------- | ---------- | --- | ----------- | --- | --- | ------------- | --------------------- | --- | --- | --- | --------------------------- | --- | --- | --- |
|           |            |     |             |     |     |               | grouppredictionsintoM |     |     |     | intervalbins(eachofsize1/M) |     |     |     |
well-calibrated,asconfidencecloselyapproximatestheex-
|                     |     |                                  |     |     |     |     | andcalculatetheaccuracyofeachbin. |     |     |     |     |     | LetB | m betheset |
| ------------------- | --- | -------------------------------- | --- | --- | --- | --- | --------------------------------- | --- | --- | --- | --- | --- | ---- | ---------- |
| pectedaccuracy(i.e. |     | thebarsalignroughlyalongthediag- |     |     |     |     |                                   |     |     |     |     |     |      |            |
ofindicesofsampleswhosepredictionconfidencefallsinto
onal). On the other hand, the ResNet’s accuracy is better, theintervalI =(m−1, m]. TheaccuracyofB is
|                               |     |     |     |     |     |     |     |     | m   |     |     |     |     | m   |
| ----------------------------- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| butdoesnotmatchitsconfidence. |     |     |     |     |     |     |     |     |     | M   | M   |     |     |     |
1
(cid:88)
|     |     |     |     |     |     |     |     |     | acc(B | )=  |     | 1(yˆ | =y  | ),  |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | ----- | --- | --- | ---- | --- | --- |
Our goal is not only to understand why neural networks m |B | i i
m
| havebecomemiscalibrated,butalsotoidentifywhatmeth- |     |     |     |     |     |     |     |     |     |     | i∈Bm |     |     |     |
| -------------------------------------------------- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | ---- | --- | --- | --- |
ods can alleviate this problem. In this paper, we demon- where yˆ and y are the predicted and true class labels for
|                                                  |     |     |     |     |     |     |          | i   | i                                |     |     |      |     |          |
| ------------------------------------------------ | --- | --- | --- | --- | --- | --- | -------- | --- | -------------------------------- | --- | --- | ---- | --- | -------- |
|                                                  |     |     |     |     |     |     | samplei. |     | Basicprobabilitytellsusthatacc(B |     |     |      |     | )isanun- |
| strateonseveralcomputervisionandNLPtasksthatneu- |     |     |     |     |     |     |          |     |                                  |     |     |      |     | m        |
|                                                  |     |     |     |     |     |     |          |     |                                  |     |     | P(Yˆ |     | Pˆ       |
ralnetworksproduceconfidencesthatdonotrepresenttrue biased and consistent estimator of = Y | ∈ I m ).
|     |     |     |     |     |     |     | WedefinetheaverageconfidencewithinbinB |     |     |     |     |     |     | as  |
| --- | --- | --- | --- | --- | --- | --- | -------------------------------------- | --- | --- | --- | --- | --- | --- | --- |
probabilities. Additionally, we offer insight and intuition m
into network training and architectural trends that may 1 (cid:88)
|     |     |     |     |     |     |     |     |     | conf(B |     | )=  |     | pˆ, |     |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | ------ | --- | --- | --- | --- | --- |
cause miscalibration. Finally, we compare various post- m |B | i
m
i∈Bm
| processing                                             | calibration   | methods |         | on state-of-the-art |     | neural      |                          |     |                   |     |               |            |                |       |
| ------------------------------------------------------ | ------------- | ------- | ------- | ------------------- | --- | ----------- | ------------------------ | --- | ----------------- | --- | ------------- | ---------- | -------------- | ----- |
|                                                        |               |         |         |                     |     |             | where                    | pˆ  | is the confidence |     | for           | sample     | i. acc(B       | ) and |
| networks,                                              | and introduce |         | several | extensions          |     | of our own. |                          | i   |                   |     |               |            |                | m     |
|                                                        |               |         |         |                     |     |             | conf(B                   | )   | approximate       |     | the left-hand | and        | right-hand     | sides |
| Surprisingly,wefindthatasingle-parametervariantofPlatt |               |         |         |                     |     |             |                          | m   |                   |     |               |            |                |       |
|                                                        |               |         |         |                     |     |             | of(1)respectivelyforbinB |     |                   |     | .             | Therefore, | aperfectlycal- |       |
scaling (Platt et al., 1999) – which we refer to as temper- m
ature scaling – is often the most effective method at ob- ibrated model will have acc(B m ) = conf(B m ) for all
m∈{1,...,M}.Notethatreliabilitydiagramsdonotdis-
| taining calibrated |     | probabilities. |     | Because | this | method is |     |     |     |     |     |     |     |     |
| ------------------ | --- | -------------- | --- | ------- | ---- | --------- | --- | --- | --- | --- | --- | --- | --- | --- |
playtheproportionofsamplesinagivenbin,andthuscan-
| straightforward | to  | implement |     | with existing | deep | learning |     |     |     |     |     |     |     |     |
| --------------- | --- | --------- | --- | ------------- | ---- | -------- | --- | --- | --- | --- | --- | --- | --- | --- |
frameworks,itcanbeeasilyadoptedinpracticalsettings. notbeusedtoestimatehowmanysamplesarecalibrated.
|     |     |     |     |     |     |     | Expected |     | Calibration |     | Error | (ECE). | While | reliability |
| --- | --- | --- | --- | --- | --- | --- | -------- | --- | ----------- | --- | ----- | ------ | ----- | ----------- |
2.Definitions
|     |     |     |     |     |     |     | diagrams |     | are useful | visual | tools, | it is | more convenient | to  |
| --- | --- | --- | --- | --- | --- | --- | -------- | --- | ---------- | ------ | ------ | ----- | --------------- | --- |
Theproblemweaddressinthispaperissupervisedmulti- haveascalarsummarystatisticofcalibration. Sincestatis-
classclassificationwithneuralnetworks.TheinputX ∈X ticscomparingtwodistributionscannotbecomprehensive,
and label Y ∈ Y = {1,...,K} are random variables previousworkshaveproposedvariants,eachwithaunique
that follow a ground truth joint distribution π(X,Y) = emphasis. Onenotionofmiscalibrationisthedifferencein
π(Y|X)π(X). Let h be a neural network with h(X) = expectationbetweenconfidenceandaccuracy,i.e.
| (Yˆ,Pˆ), | Yˆ  |     |     |     | Pˆ  |     |     |     |     |     |     |     |     |     |
| -------- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
where is a class prediction and is its associ- (cid:104)(cid:12) (cid:16) (cid:17) (cid:12)(cid:105)
|     |     |     |     |     |     |     |     |     | E (cid:12)P | Yˆ =Y | |Pˆ | =p  | −p(cid:12) | (2) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | ----------- | ----- | --- | --- | ---------- | --- |
atedconfidence,i.e. probabilityofcorrectness. Wewould (cid:12) (cid:12)
Pˆ
Pˆ
| like the confidence |     | estimate |     | to be calibrated, |     | which in- |     |     |     |     |     |     |     |     |
| ------------------- | --- | -------- | --- | ----------------- | --- | --------- | --- | --- | --- | --- | --- | --- | --- | --- |
ExpectedCalibrationError(Naeinietal.,2015)–orECE
| tuitively means | that | Pˆ  | represents | a true | probability. | For |                |     |     |     |              |             |     |        |
| --------------- | ---- | --- | ---------- | ------ | ------------ | --- | -------------- | --- | --- | --- | ------------ | ----------- | --- | ------ |
|                 |      |     |            |        |              |     | – approximates |     | (2) | by  | partitioning | predictions |     | into M |
example, given 100 predictions, each with confidence of equally-spacedbins(similartothereliabilitydiagrams)and

0.7
0.6
0.5
0.4
0.3
0.2
0.1
0.0
0 20 40 60 80 100120
Depth
ECE/rorrE
VaryingDepth VaryingWidth UsingNormalization VaryingWeightDecay
ResNet-CIFAR-100 ResNet-14-CIFAR-100 ConvNet-CIFAR-100 ResNet-110-CIFAR-100
Error Error Error Error
ECE ECE ECE ECE
0 50 100150200250300 Without With 10 − 5 10 − 4 10 − 3 10 − 2
Filtersperlayer BatchNormalization Weightdecay
Figure2.Theeffectofnetworkdepth(farleft),width(middleleft),BatchNormalization(middleright),andweightdecay(farright)on
miscalibration,asmeasuredbyECE(lowerisbetter).
takingaweightedaverageofthebins’accuracy/confidence 3.ObservingMiscalibration
difference. Moreprecisely,
The architecture and training procedures of neural net-
M (cid:12) (cid:12)
ECE= (cid:88) |B n m |(cid:12) (cid:12) (cid:12) acc(B m )−conf(B m ) (cid:12) (cid:12) (cid:12) , (3) w tio o n rk w s e ha id v e e n r ti a f p y id s l o y m e e vo re lv c e e d nt i c n h r a e n c g e e n s t t y h e a a t r a s r . e I r n es t p h o i n s s s ib ec le -
m=1
for the miscalibration phenomenon observed in Figure 1.
wherenisthenumberofsamples. Thedifferencebetween
Though we cannot claim causality, we find that increased
accandconf foragivenbinrepresentsthecalibrationgap
model capacity and lack of regularization are closely re-
(red bars in reliability diagrams – e.g. Figure 1). We use
latedtomodelmiscalibration.
ECE as the primary empirical metric to measure calibra-
tion. SeeSectionS1formoreanalysisofthismetric.
Modelcapacity. Themodelcapacityofneuralnetworks
has increased at a dramatic pace over the past few years.
Maximum Calibration Error (MCE). In high-risk ap-
It is now common to see networks with hundreds, if not
plications where reliable confidence measures are abso-
thousands of layers (He et al., 2016; Huang et al., 2016)
lutelynecessary,wemaywishtominimizetheworst-case
andhundredsofconvolutionalfiltersperlayer(Zagoruyko
deviationbetweenconfidenceandaccuracy:
(cid:12) (cid:16) (cid:17) (cid:12) & Komodakis, 2016). Recent work shows that very deep
max (cid:12) (cid:12) P Yˆ =Y |Pˆ =p −p(cid:12) (cid:12) . (4) or wide models are able to generalize better than smaller
p∈[0,1]
ones,whileexhibitingthecapacitytoeasilyfitthetraining
TheMaximumCalibrationError(Naeinietal.,2015)–or
set(Zhangetal.,2017).
MCE–estimatesthisdeviation. SimilarlytoECE,thisap-
proximationinvolvesbinning: Although increasing depth and width may reduce classi-
fication error, we observe that these increases negatively
MCE= max |acc(B )−conf(B )|. (5)
m m
m∈{1,...,M} affectmodelcalibration. Figure2displayserrorandECE
We can visualize MCE and ECE on reliability diagrams. as a function of depth and width on a ResNet trained on
MCEisthelargestcalibrationgap(redbars)acrossallbins, CIFAR-100. Thefarleftfigurevariesdepthforanetwork
whereas ECE is a weighted average of all gaps. For per- with64convolutionalfiltersperlayer,whilethemiddleleft
fectlycalibratedclassifiers,MCEandECEbothequal0. figure fixes the depth at 14 layers and varies the number
of convolutional filters per layer. Though even the small-
estmodelsinthegraphexhibitsomedegreeofmiscalibra-
Negativeloglikelihood isastandardmeasureofaprob-
tion, the ECE metric grows substantially with model ca-
abilisticmodel’squality(Friedmanetal.,2001). Itisalso
pacity. Duringtraining,afterthemodelisabletocorrectly
referredtoasthecrossentropylossinthecontextofdeep
classify (almost) all training samples, NLL can be further
learning(Bengioetal.,2015). Givenaprobabilisticmodel
minimizedbyincreasingtheconfidenceofpredictions. In-
πˆ(Y|X)andnsamples,NLLisdefinedas:
creased model capacity will lower training NLL, and thus
n
(cid:88) themodelwillbemore(over)confidentonaverage.
L=− log(πˆ(y |x )) (6)
i i
i=1
Itisastandardresult(Friedmanetal.,2001)that,inexpec- BatchNormalization (Ioffe&Szegedy,2015)improves
tation, NLL is minimized if and only if πˆ(Y|X) recovers the optimization of neural networks by minimizing distri-
thegroundtruthconditionaldistributionπ(Y|X). butionshiftsinactivationswithintheneuralnetwork’shid-

45
40
35
30
25
20
0 100 200 300 400 500
Epoch
)delacs(
LLN
/
)%(
rorrE
NLL Overfitting on CIFAR−100 plays training error and ECE for a 110-layer ResNet with
varying amounts of weight decay. The only other forms
Test error
Test NLL ofregularizationaredataaugmentationandBatchNormal-
ization. We observe that calibration and accuracy are not
optimizedbythesameparametersetting. Whilethemodel
exhibits both over-regularization and under-regularization
with respect to classification error, it does not appear that
calibration is negatively impacted by having too much
weight decay. Model calibration continues to improve
when more regularization is added, well after the point of
achievingoptimalaccuracy. Theslightuptickattheendof
thegraphmaybeanartifactofusingaweightdecayfactor
thatimpedesoptimization.
NLL can be used to indirectly measure model calibra-
tion. In practice, we observe a disconnect between NLL
Figure3.TesterrorandNLLofa110-layerResNetwithstochas-
andaccuracy,whichmayexplainthemiscalibrationinFig-
ticdepthonCIFAR-100duringtraining.NLLisscaledbyacon-
ure2. Thisdisconnectoccursbecauseneuralnetworkscan
stanttofitinthefigure.Learningratedropsby10xatepochs250
and375.Theshadedareamarksbetweenepochsatwhichthebest overfit to NLL without overfitting to the 0/1 loss. We ob-
validationlossandbestvalidationerrorareproduced. servethistrendinthetrainingcurvesofsomemiscalibrated
models. Figure 3 shows test error and NLL (rescaled to
den layers. Recent research suggests that these normal- match error) on CIFAR-100 as training progresses. Both
ization techniques have enabled the development of very error and NLL immediately drop at epoch 250, when the
deep architectures, such as ResNets (He et al., 2016) and learningrateisdropped;however,NLLoverfitsduringthe
DenseNets (Huang et al., 2017). It has been shown that remainder of training. Surprisingly, overfitting to NLL is
Batch Normalization improves training time, reduces the beneficial to classification accuracy. On CIFAR-100, test
need for additional regularization, and can in some cases error drops from 29% to 27% in the region where NLL
improvetheaccuracyofnetworks. overfits. Thisphenomenonrendersaconcreteexplanation
of miscalibration: the network learns better classification
WhileitisdifficulttopinpointexactlyhowBatchNormal-
accuracyattheexpenseofwell-modeledprobabilities.
ization affects the final predictions of a model, we do ob-
servethatmodelstrainedwithBatchNormalizationtendto Wecanconnectthisfindingtorecentworkexaminingthe
bemoremiscalibrated. InthemiddlerightplotofFigure2, generalizationoflargeneuralnetworks.Zhangetal.(2017)
we see that a 6-layer ConvNet obtains worse calibration observe that deep neural networks seemingly violate the
when Batch Normalization is applied, even though classi- commonunderstandingoflearningtheorythatlargemod-
ficationaccuracyimprovesslightly. Wefindthatthisresult els with little regularization will not generalize well. The
holdsregardlessofthehyperparametersusedontheBatch observed disconnect between NLL and 0/1 loss suggests
Normalizationmodel(i.e. loworhighlearningrate,etc.). thatthesehighcapacitymodelsarenotnecessarilyimmune
fromoverfitting,butrather,overfittingmanifestsinproba-
Weight decay, which used to be the predominant regu- bilisticerrorratherthanclassificationerror.
larization mechanism for neural networks, is decreasingly
utilized when training modern neural networks. Learning 4.CalibrationMethods
theory suggests that regularization is necessary to prevent
overfitting,especiallyasmodelcapacityincreases(Vapnik, In this section, we first review existing calibration meth-
1998). However,duetotheapparentregularizationeffects ods, and introduce new variants of our own. All methods
of Batch Normalization, recent research seems to suggest are post-processing steps that produce (calibrated) proba-
that models with less L2 regularization tend to generalize bilities. Each method requires a hold-out validation set,
better(Ioffe&Szegedy,2015). Asaresult,itisnowcom- whichinpracticecanbethesamesetusedforhyperparam-
mon to train models with little weight decay, if any at all. eter tuning. We assume that the training, validation, and
ThetopperformingImageNetmodelsof2015alluseanor- testsetsaredrawnfromthesamedistribution.
derofmagnitudelessweightdecaythanmodelsofprevious
years(Heetal.,2016;Simonyan&Zisserman,2015). 4.1.CalibratingBinaryModels
Wefindthattrainingwithlessweightdecayhasanegative We first introduce calibration in the binary setting, i.e.
impact on calibration. The far right plot in Figure 2 dis- Y = {0,1}. For simplicity, throughout this subsection,

we assume the model outputs only the confidence for the model averaging. Essentially, BBQ marginalizes out all
positiveclass.1 Givenasamplex ,wehaveaccesstopˆ – possiblebinningschemestoproduceqˆ. Moreformally, a
i i i
the network’s predicted probability of y = 1, as well as binningschemesisapair(M,I)whereM isthenumber
i
z ∈ R – which is the network’s non-probabilistic output, ofbins,andI isacorrespondingpartitioningof[0,1]into
i
orlogit. Thepredictedprobabilitypˆ isderivedfromz us- disjointintervals(0 = a ≤ a ≤ ... ≤ a = 1). The
i i 1 2 M+1
ing a sigmoid function σ; i.e. pˆ = σ(z ). Our goal is to parametersofabinningschemeareθ ,...,θ .Underthis
i i 1 M
produceacalibratedprobabilityqˆ basedony ,pˆ,andz . framework,histogrambinningandisotonicregressionboth
i i i i
produceasinglebinningscheme,whereasBBQconsiders
Histogrambinning (Zadrozny&Elkan,2001)isasim- a space S of all possible binning schemes for the valida-
ple non-parametric calibration method. In a nutshell, all tion dataset D. BBQ performs Bayesian averaging of the
uncalibrated predictions pˆ are divided into mutually ex-
probabilitiesproducedbyeachscheme:2
i
clusivebinsB 1 ,...,B M . Eachbinisassignedacalibrated P(qˆ te |pˆ te ,D)= (cid:88) P(qˆ te ,S =s|pˆ te ,D)
scoreθ ;i.e. ifpˆ isassignedtobinB ,thenqˆ =θ . At
m i m i m s∈S
testtime,ifpredictionpˆ te fallsintobinB m ,thenthecali- = (cid:88) P(qˆ |pˆ ,S=s,D)P(S=s|D).
bratedpredictionqˆ isθ . Moreprecisely, forasuitably te te
te m
chosen M (usually small), we first define bin boundaries s∈S
0 = a ≤ a ≤ ... ≤ a = 1, where the bin B where P(qˆ te | pˆ te ,S =s,D) is the calibrated probability
1 2 M+1 m
is defined by the interval (a ,a ]. Typically the bin usingbinningschemes. Usingauniformprior,theweight
m m+1
P(S=s|D)canbederivedusingBayes’rule:
boundariesareeitherchosentobeequallengthintervalsor
toequalizethenumberofsamplesineachbin. Thepredic- P(D |S=s)
P(S=s|D)= .
tionsθ i arechosentominimizethebin-wisesquaredloss: (cid:80) s(cid:48)∈S P(D |S=s(cid:48))
Theparametersθ ,...,θ canbeviewedasparametersof
1 M
M n
min (cid:88) (cid:88) 1(a ≤pˆ <a )(θ −y )2, (7) M independent binomialdistributions. Hence, byplacing
m i m+1 m i a Beta prior on θ ,...,θ , we can obtain a closed form
θ1,...,θM
m=1i=1 expressionforthe
1
margin
M
allikelihoodP(D | S=s). This
where1istheindicatorfunction. Givenfixedbinsbound-
allowsustocomputeP(qˆ |pˆ ,D)foranytestinput.
te te
aries,thesolutionto(7)resultsinθ thatcorrespondtothe
m
averagenumberofpositive-classsamplesinbinB .
m
Plattscaling (Plattetal.,1999)isaparametricapproach
to calibration, unlike the other approaches. The non-
Isotonicregression (Zadrozny&Elkan,2002),arguably
probabilisticpredictionsofaclassifierareusedasfeatures
the most common non-parametric calibration method,
foralogisticregressionmodel,whichistrainedontheval-
learns a piecewise constant function f to transform un-
idationsettoreturnprobabilities. Morespecifically,inthe
calibrated outputs; i.e. qˆ = f(pˆ). Specifically, iso-
i i context of neural networks (Niculescu-Mizil & Caruana,
tonic regression produces f to minimize the square loss 2005), Platt scaling learns scalar parameters a,b ∈ R and
(cid:80)n (f(pˆ)−y )2. Becausef isconstrainedtobepiece-
i=1 i i outputsqˆ i = σ(az i +b)asthecalibratedprobability. Pa-
wise constant, we can write the optimization problem as:
rametersaandbcanbeoptimizedusingtheNLLlossover
the validation set. It is important to note that the neural
M n
min (cid:88) (cid:88) 1(a ≤pˆ <a )(θ −y )2 network’sparametersarefixedduringthisstage.
m i m+1 m i
M
a1 θ , 1 . , . . . . , . a ,θ M M +1 m=1i=1 4.2.ExtensiontoMulticlassModels
subjectto 0=a ≤a ≤...≤a =1,
1 2 M+1 For classification problems involving K > 2 classes, we
θ 1 ≤θ 2 ≤...≤θ M . return to the original problem formulation. The network
whereM isthenumberofintervals;a 1 ,...,a M+1 arethe outputs a class prediction yˆ i and confidence score pˆ i for
interval boundaries; and θ 1 ,...,θ M are the function val- eachinputx i .Inthiscase,thenetworklogitsz i arevectors,
ues. Under this parameterization, isotonic regression is a whereyˆ i =argmax k z i (k),andpˆ i istypicallyderivedusing
strictgeneralizationofhistogrambinninginwhichthebin thesoftmaxfunctionσ SM :
boundariesandbinpredictionsarejointlyoptimized. exp(z(k))
σ (z )(k) = i , pˆ =max σ (z )(k).
SM i (cid:80)K exp(z(j)) i k SM i
BayesianBinningintoQuantiles(BBQ) (Naeinietal., j=1 i
2015) is a extension of histogram binning using Bayesian Thegoalistoproduceacalibratedconfidenceqˆ i and(pos-
siblynew)classpredictionyˆ(cid:48) basedony ,yˆ,pˆ,andz .
i i i i i
1ThisisincontrastwiththesettinginSection2,inwhichthe
modelproducesbothaclasspredictionandconfidence. 2Becausethevalidationdatasetisfinite,Sisaswell.

Dataset Model Uncalibrated Hist.Binning Isotonic BBQ Temp.Scaling VectorScaling MatrixScaling
Birds ResNet50 9.19% 4.34% 5.22% 4.12% 1.85% 3.0% 21.13%
Cars ResNet50 4.3% 1.74% 4.29% 1.84% 2.35% 2.37% 10.5%
CIFAR-10 ResNet110 4.6% 0.58% 0.81% 0.54% 0.83% 0.88% 1.0%
CIFAR-10 ResNet110(SD) 4.12% 0.67% 1.11% 0.9% 0.6% 0.64% 0.72%
CIFAR-10 WideResNet32 4.52% 0.72% 1.08% 0.74% 0.54% 0.6% 0.72%
CIFAR-10 DenseNet40 3.28% 0.44% 0.61% 0.81% 0.33% 0.41% 0.41%
CIFAR-10 LeNet5 3.02% 1.56% 1.85% 1.59% 0.93% 1.15% 1.16%
CIFAR-100 ResNet110 16.53% 2.66% 4.99% 5.46% 1.26% 1.32% 25.49%
CIFAR-100 ResNet110(SD) 12.67% 2.46% 4.16% 3.58% 0.96% 0.9% 20.09%
CIFAR-100 WideResNet32 15.0% 3.01% 5.85% 5.77% 2.32% 2.57% 24.44%
CIFAR-100 DenseNet40 10.37% 2.68% 4.51% 3.59% 1.18% 1.09% 21.87%
CIFAR-100 LeNet5 4.85% 6.48% 2.35% 3.77% 2.02% 2.09% 13.24%
ImageNet DenseNet161 6.28% 4.52% 5.18% 3.51% 1.99% 2.24% -
ImageNet ResNet152 5.48% 4.36% 4.77% 3.56% 1.86% 2.23% -
SVHN ResNet152(SD) 0.44% 0.14% 0.28% 0.22% 0.17% 0.27% 0.17%
20News DAN3 8.02% 3.6% 5.52% 4.98% 4.11% 4.61% 9.1%
Reuters DAN3 0.85% 1.75% 1.15% 0.97% 0.91% 0.66% 1.58%
SSTBinary TreeLSTM 6.63% 1.93% 1.65% 2.27% 1.84% 1.84% 1.84%
SSTFineGrained TreeLSTM 6.71% 2.09% 1.65% 2.61% 2.56% 2.98% 2.39%
Table1.ECE(%)(withM = 15bins)onstandardvisionandNLPdatasetsbeforecalibrationandwithvariouscalibrationmethods.
Thenumberfollowingamodel’snamedenotesthenetworkdepth.
Extensionofbinningmethods. Onecommonwayofex- T is called the temperature, and it “softens” the softmax
tendingbinarycalibrationmethodstothemulticlasssetting (i.e. raises the output entropy) with T > 1. As T → ∞,
is by treating the problem as K one-versus-all problems theprobabilityqˆ approaches1/K,whichrepresentsmax-
i
(Zadrozny&Elkan,2002). Fork = 1,...,K, weforma imum uncertainty. With T = 1, we recover the original
binary calibration problem where the label is 1(y = k) probability pˆ. As T → 0, the probability collapses to a
i i
and the predicted probability is σ (z )(k). This gives point mass (i.e. qˆ = 1). T is optimized with respect to
SM i i
us K calibration models, each for a particular class. At NLL onthe validationset. Because theparameter T does
test time, we obtain an unnormalized probability vector notchangethemaximumofthesoftmaxfunction,theclass
[qˆ(1),...,qˆ(K)],whereqˆ(k)isthecalibratedprobabilityfor predictionyˆ(cid:48) remainsunchanged. Inotherwords, temper-
i i i i
class k. The new class prediction yˆ(cid:48) is the argmax of the aturescalingdoesnotaffectthemodel’saccuracy.
i
vector, andthenewconfidenceqˆ(cid:48) isthemaxofthevector
i Temperaturescalingiscommonlyusedinsettingssuchas
normalized by (cid:80)K k=1 qˆ i (k). This extension can be applied knowledge distillation (Hinton et al., 2015) and statistical
tohistogrambinning,isotonicregression,andBBQ. mechanics (Jaynes, 1957). To the best of our knowledge,
wearenotawareofanyprioruseinthecontextofcalibrat-
Matrix and vector scaling are two multi-class exten- ingprobabilisticmodels.3 Themodelisequivalenttomax-
sionsofPlattscaling. Letz bethelogitsvectorproduced imizing the entropy of the output probability distribution
i
before the softmax layer for input x . Matrix scaling ap- subjecttocertainconstraintsonthelogits(seeSectionS2).
i
pliesalineartransformationWz +btothelogits:
i
4.3.OtherRelatedWorks
qˆ =max σ (Wz +b)(k),
i SM i
k
(8) Calibrationandconfidencescoreshavebeenstudiedinvar-
yˆ(cid:48) =argmax(Wz +b)(k).
i i ious contexts in recent years. Kuleshov & Ermon (2016)
k
studytheproblemofcalibrationintheonlinesetting,where
The parameters W and b are optimized with respect to
theinputscancomefromapotentiallyadversarialsource.
NLL on the validation set. As the number of parameters
Kuleshov & Liang (2015) investigate how to produce cal-
formatrixscalinggrowsquadraticallywiththenumberof
ibrated probabilities when the output space is a structured
classesK,wedefinevectorscalingasavariantwhereW
object. Lakshminarayanan et al. (2016) use ensembles of
isrestrictedtobeadiagonalmatrix.
networks to obtain uncertainty estimates. Pereyra et al.
(2017)penalizeoverconfidentpredictionsasaformofreg-
Temperature scaling, the simplest extension of Platt
ularization. Hendrycks & Gimpel (2017) use confidence
scaling,usesasinglescalarparameterT >0forallclasses.
Giventhelogitvectorz ,thenewconfidencepredictionis 3Tohighlighttheconnectionwithpriorworkswedefinetem-
i
peraturescalingintermsof 1 insteadofamultiplicativescalar.
qˆ =max σ (z /T)(k). (9) T
i SM i
k

scorestodetermineifsamplesareout-of-distribution. gories by content. 9034/2259/7528 documents for
train/validation/test.
| Bayesian   | neural      | networks |              | (Denker &    | Lecun,       | 1990;     |             |     |        |                |             |     |           |         |
| ---------- | ----------- | -------- | ------------ | ------------ | ------------ | --------- | ----------- | --- | ------ | -------------- | ----------- | --- | --------- | ------- |
|            |             |          |              |              |              |           | 2. Reuters: |     | News   | articles,      | partitioned |     | into      | 8 cate- |
| MacKay,    | 1992)       | return a | probability  | distribution |              | over out- |             |     |        |                |             |     |           |         |
|            |             |          |              |              |              |           | gories      | by  | topic. | 4388/1097/2189 |             |     | documents | for     |
| puts as an | alternative | way      | to represent | model        | uncertainty. |           |             |     |        |                |             |     |           |         |
train/validation/test.
| Gal & Ghahramani |               | (2016)          | draw       | a connection |              | between  |                                   |       |           |          |             |               |          |         |
| ---------------- | ------------- | --------------- | ---------- | ------------ | ------------ | -------- | --------------------------------- | ----- | --------- | -------- | ----------- | ------------- | -------- | ------- |
|                  |               |                 |            |              |              |          | 3. Stanford                       |       | Sentiment | Treebank |             | (SST)         | (Socher  | et al., |
| Dropout          | (Srivastava   | et              | al., 2014) | and model    | uncertainty, |          |                                   |       |           |          |             |               |          |         |
|                  |               |                 |            |              |              |          | 2013):                            | Movie | reviews,  |          | represented | as            | sentence | parse   |
| claiming         | that sampling |                 | models     | with dropped | nodes        | is a     |                                   |       |           |          |             |               |          |         |
|                  |               |                 |            |              |              |          | treesthatareannotatedbysentiment. |       |           |          |             | Eachsamplein- |          |         |
| way to estimate  |               | the probability |            | distribution | over         | all pos- |                                   |       |           |          |             |               |          |         |
cludesacoarsebinarylabelandafinegrained5-class
| sible models | for | a given | sample. | Kendall | & Gal | (2017) |        |     |           |     |         |             |     |        |
| ------------ | --- | ------- | ------- | ------- | ----- | ------ | ------ | --- | --------- | --- | ------- | ----------- | --- | ------ |
|              |     |         |         |         |       |        | label. | As  | described | in  | (Tai et | al., 2015), | the | train- |
combinethisapproachwithamodelthatoutputsapredic-
|                                      |     |     |     |     |              |     | ing/validation/test |     |     | sets | contain | 6920/872/1821 |     | docu- |
| ------------------------------------ | --- | --- | --- | --- | ------------ | --- | ------------------- | --- | --- | ---- | ------- | ------------- | --- | ----- |
| tivemeanandvarianceforeachdatapoint. |     |     |     |     | Thisnotionof |     |                     |     |     |      |         |               |     |       |
mentsforbinary,and544/1101/2210forfine-grained.
| uncertaintyisnotrestrictedtoclassificationproblems. |     |     |     |     |     | Ad- |     |     |     |     |     |     |     |     |
| --------------------------------------------------- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
ditionally,neuralnetworkscanbeusedinconjunctionwith On 20 News and Reuters, we train Deep Averaging Net-
Bayesian models that output complete distributions. For works (DANs) (Iyyer et al., 2015) with 3 feed-forward
example,deepkernellearning(Wilsonetal.,2016a;b;Al- layers and Batch Normalization. On SST, we train
Shedivatetal.,2016)combinesdeepneuralnetworkswith TreeLSTMs(LongShortTermMemory)(Taietal.,2015).
Gaussian processes on classification and regression prob- For both models we use the default hyperparmaeters sug-
lems. Incontrast,ourframework,whichdoesnotaugment gestedbytheauthors.
theneuralnetworkmodel,returnsaconfidencescorerather
thanreturningadistributionofpossibleoutputs.
|     |     |     |     |     |     |     | CalibrationResults. |     |     | Table1displaysmodelcalibration, |        |        |        |         |
| --- | --- | --- | --- | --- | --- | --- | ------------------- | --- | --- | ------------------------------- | ------ | ------ | ------ | ------- |
|     |     |     |     |     |     |     | as measured         | by  | ECE | (with                           | M = 15 | bins), | before | and af- |
5.Results
terapplyingthevariousmethods(seeSectionS3forMCE,
We apply the calibration methods in Section 4 to image NLL,anderrortables).Itisworthnotingthatmostdatasets
classificationanddocumentclassificationneuralnetworks. andmodelsexperiencesomedegreeofmiscalibration,with
4 10%.
Forimageclassificationweuse6datasets: ECE typically between to This is not architecture
|     |     |     |     |     |     |     | specific: | we observe |     | miscalibration |     | on convolutional |     | net- |
| --- | --- | --- | --- | --- | --- | --- | --------- | ---------- | --- | -------------- | --- | ---------------- | --- | ---- |
1. Caltech-UCSD Birds (Welinder et al., 2010): works (with and without skip connections), recurrent net-
| 200 | bird | species. | 5994/2897/2897 |     | images | for |                                 |     |     |     |     |                  |     |     |
| --- | ---- | -------- | -------------- | --- | ------ | --- | ------------------------------- | --- | --- | --- | --- | ---------------- | --- | --- |
|     |      |          |                |     |        |     | works,anddeepaveragingnetworks. |     |     |     |     | Thetwonotableex- |     |     |
train/validation/testsets. ceptionsareSVHNandReuters,bothofwhichexperience
2. Stanford Cars (Krause et al., 2013): 196 classes of ECE values below 1%. Both of these datasets have very
cars by make, model, and year. 8041/4020/4020 im- low error (1.98% and 2.97%, respectively); and therefore
agesfortrain/validation/test. theratioofECEtoerroriscomparabletootherdatasets.
3. ImageNet2012(Dengetal.,2009):Naturalsceneim-
|      |      |               |     |                           |     |     | Our most | important |     | discovery | is the | surprising | effective- |     |
| ---- | ---- | ------------- | --- | ------------------------- | --- | --- | -------- | --------- | --- | --------- | ------ | ---------- | ---------- | --- |
| ages | from | 1000 classes. |     | 1.3 million/25,000/25,000 |     |     |          |           |     |           |        |            |            |     |
nessoftemperaturescalingdespiteitsremarkablesimplic-
imagesfortrain/validation/test.
|     |     |     |     |     |     |     | ity. Temperaturescalingoutperformsallothermethodson |     |     |     |     |     |     |     |
| --- | --- | --- | --- | --- | --- | --- | --------------------------------------------------- | --- | --- | --- | --- | --- | --- | --- |
4. CIFAR-10/CIFAR-100(Krizhevsky&Hinton,2009):
thevisiontasks,andperformscomparablytoothermethods
| Color | images | (32 | ×   | 32) from | 10/100 | classes. |            |           |     |      |            |      |      |          |
| ----- | ------ | --- | --- | -------- | ------ | -------- | ---------- | --------- | --- | ---- | ---------- | ---- | ---- | -------- |
|       |        |     |     |          |        |          | on the NLP | datasets. |     | What | is perhaps | even | more | surpris- |
45,000/5,000/10,000imagesfortrain/validation/test.
ingisthattemperaturescalingoutperformsthevectorand
| 5. Street | View  | House   | Numbers | (SVHN) | (Netzer | et al., |               |         |           |        |         |              |             |      |
| --------- | ----- | ------- | ------- | ------ | ------- | ------- | ------------- | ------- | --------- | ------ | ------- | ------------ | ----------- | ---- |
|           |       |         |         |        |         |         | matrix Platt  | scaling | variants, |        | which   | are strictly | more        | gen- |
| 2011):    | 32    | × 32    | colored | images | of      | cropped |               |         |           |        |         |              |             |      |
|           |       |         |         |        |         |         | eral methods. |         | In fact,  | vector | scaling | recovers     | essentially |      |
| out       | house | numbers | from    | Google | Street  | View.   |               |         |           |        |         |              |             |      |
thesamesolutionastemperaturescaling–thelearnedvec-
598,388/6,000/26,032imagesfortrain/validation/test.
torhasnearlyconstantentries,andthereforeisnodifferent
We train state-of-the-art convolutional networks: ResNets thanascalartransformation. Inotherwords,networkmis-
calibrationisintrinsicallylowdimensional.
| (He et al., | 2016), | ResNets     | with    | stochastic | depth | (SD)  |     |     |     |     |     |     |     |     |
| ----------- | ------ | ----------- | ------- | ---------- | ----- | ----- | --- | --- | --- | --- | --- | --- | --- | --- |
| (Huang et   | al.,   | 2016), Wide | ResNets | (Zagoruyko |       | & Ko- |     |     |     |     |     |     |     |     |
Theonlydatasetthattemperaturescalingdoesnotcalibrate
| modakis, | 2016), | and DenseNets |     | (Huang et | al., 2017). | We  |                |     |          |         |           |      |     |        |
| -------- | ------ | ------------- | --- | --------- | ----------- | --- | -------------- | --- | -------- | ------- | --------- | ---- | --- | ------ |
|          |        |               |     |           |             |     | is the Reuters |     | dataset. | In this | instance, | only | one | of the |
usethedatapreprocessing,trainingprocedures,andhyper- abovemethodsisabletoimprovecalibration. Becausethis
| parametersasdescribedineachpaper. |     |     |     | ForBirdsandCars, |     |     |     |     |     |     |     |     |     |     |
| --------------------------------- | --- | --- | --- | ---------------- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
datasetiswell-calibratedtobeginwith(ECE≤1%),there
wefine-tunenetworkspretrainedonImageNet. is not much room for improvement with any method, and
Fordocumentclassificationweexperimentwith4datasets: post-processing may not even be necessary to begin with.
|     |     |     |     |     |     |     | It is also | possible | that | our | measurements |     | are affected | by  |
| --- | --- | --- | --- | --- | --- | --- | ---------- | -------- | ---- | --- | ------------ | --- | ------------ | --- |
1. 20 News: News articles, partitioned into 20 cate- datasetsplitorbytheparticularbinningscheme.

Uncal.-CIFAR-100 Temp.Scale-CIFAR-100 Hist.Bin.-CIFAR-100 Iso.Reg.-CIFAR-100
|     | ResNet-110(SD) |     |     |     | ResNet-110(SD) |     |     | ResNet-110(SD) |     |     |     | ResNet-110(SD) |     |     |     |
| --- | -------------- | --- | --- | --- | -------------- | --- | --- | -------------- | --- | --- | --- | -------------- | --- | --- | --- |
1.0
|     |     | Outputs |     |     | Outputs |     |     |     | Outputs |     |     |     | Outputs |     |     |
| --- | --- | ------- | --- | --- | ------- | --- | --- | --- | ------- | --- | --- | --- | ------- | --- | --- |
| 0.8 |     | Gap     |     |     | Gap     |     |     |     | Gap     |     |     |     | Gap     |     |     |
ycaruccA
0.6
0.4
0.2
|     |     | ECE=12.67 |     |     |     | ECE=0.96 |     |     | ECE=2.46 |     |     |     | ECE=4.16 |     |     |
| --- | --- | --------- | --- | --- | --- | -------- | --- | --- | -------- | --- | --- | --- | -------- | --- | --- |
0.0
0.0 0.2 0.4 0.6 0.8 1.0 0.0 0.2 0.4 0.6 0.8 1.0 0.0 0.2 0.4 0.6 0.8 1.0 0.0 0.2 0.4 0.6 0.8 1.0
Confidence
Figure4. ReliabilitydiagramsforCIFAR-100before(farleft)andaftercalibration(middleleft,middleright,farright).
Matrixscalingperformspoorlyondatasetswithhundreds computationalcomplexityofvectorandmatrixscalingare
of classes (i.e. Birds, Cars, and CIFAR-100), and fails linearandquadraticrespectivelyinthenumberofclasses,
to converge on the 1000-class ImageNet dataset. This is reflecting the number of parameters in each method. For
expected, since the number of parameters scales quadrat- CIFAR-100(K =100),findinganear-optimalvectorscal-
ically with the number of classes. Any calibration model ing solution with conjugate gradient descent requires at
withtensofthousands(ormore)parameterswilloverfitto least2ordersofmagnitudemoretime. Histogrambinning
asmallvalidationset,evenwhenapplyingregularization. and isotonic regression take an order of magnitude longer
thantemperaturescaling,andBBQtakesroughly3orders
Binningmethodsimprovecalibrationonmostdatasets,but
ofmagnitudemoretime.
| do not | outperform | temperature |     | scaling. | Additionally, | bin- |     |     |     |     |     |     |     |     |     |
| ------ | ---------- | ----------- | --- | -------- | ------------- | ---- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
ningmethodstendtochangeclasspredictionswhichhurts
accuracy(seeSectionS3).Histogrambinning,thesimplest
|         |         |           |             |     |          |            | Easeofimplementation. |     |     |     | BBQisarguablythemostdif- |     |     |     |     |
| ------- | ------- | --------- | ----------- | --- | -------- | ---------- | --------------------- | --- | --- | --- | ------------------------ | --- | --- | --- | --- |
| binning | method, | typically | outperforms |     | isotonic | regression |                       |     |     |     |                          |     |     |     |     |
and BBQ, despite the fact that both methods are strictly ficult to implement, as it requires implementing a model
|               |     |              |          |     |         |            | averaging |               | scheme. | While       | all | other methods |     | are relatively |     |
| ------------- | --- | ------------ | -------- | --- | ------- | ---------- | --------- | ------------- | ------- | ----------- | --- | ------------- | --- | -------------- | --- |
| more general. |     | This further | supports | our | finding | that cali- |           |               |         |             |     |               |     |                |     |
|               |     |              |          |     |         |            | easy      | to implement, |         | temperature |     | scaling       | may | arguably       | be  |
brationisbestcorrectedbysimplemodels.
|             |           |     |        |            |             |      | the    | most      | straightforward |             | to incorporate |         | into    | a neural     | net- |
| ----------- | --------- | --- | ------ | ---------- | ----------- | ---- | ------ | --------- | --------------- | ----------- | -------------- | ------- | ------- | ------------ | ---- |
|             |           |     |        |            |             |      | work   | pipeline. |                 | In Torch7   | (Collobert     |         | et al., | 2011), for   | ex-  |
| Reliability | diagrams. |     | Figure | 4 contains | reliability | dia- |        |           |                 |             |                |         |         |              |      |
|             |           |     |        |            |             |      | ample, | we        | implement       | temperature |                | scaling |         | by inserting | a    |
gramsfor110-layerResNetsonCIFAR-100beforeandaf-
|                  |        |          |          |               |     |              | nn.MulConstant       |     |     | between |                            | the logits | and | the softmax, |     |
| ---------------- | ------ | -------- | -------- | ------------- | --- | ------------ | -------------------- | --- | --- | ------- | -------------------------- | ---------- | --- | ------------ | --- |
| ter calibration. |        | From the | far left | diagram,      | we  | see that the |                      |     |     |         |                            |            |     |              |     |
|                  |        |          |          |               |     |              | whoseparameteris1/T. |     |     |         | WesetT=1duringtraining,and |            |     |              |     |
| uncalibrated     | ResNet | tends    | to be    | overconfident |     | in its pre-  |                      |     |     |         |                            |            |     |              |     |
subsequentlyfinditsoptimalvalueonthevalidationset.4
| dictions. | We  | then can | observe | the effects | of  | temperature |     |     |     |     |     |     |     |     |     |
| --------- | --- | -------- | ------- | ----------- | --- | ----------- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
scaling(middleleft),histogrambinning(middleright),and
| isotonicregression(farright)oncalibration. |     |     |     |     |     | Allthreedis- | 6.Conclusion |     |     |     |     |     |     |     |     |
| ------------------------------------------ | --- | --- | --- | --- | --- | ------------ | ------------ | --- | --- | --- | --- | --- | --- | --- | --- |
playedmethodsproducemuchbetterconfidenceestimates.
|     |     |     |     |     |     |     | Modern |     | neural | networks | exhibit | a strange |     | phenomenon: |     |
| --- | --- | --- | --- | --- | --- | --- | ------ | --- | ------ | -------- | ------- | --------- | --- | ----------- | --- |
Ofthethreemethods,temperaturescalingmostcloselyre-
probabilisticerrorandmiscalibrationworsenevenasclas-
| coversthedesireddiagonalfunction. |     |     |     | Eachofthebinsare |     |     |            |     |       |             |     |         |              |     |      |
| --------------------------------- | --- | --- | --- | ---------------- | --- | --- | ---------- | --- | ----- | ----------- | --- | ------- | ------------ | --- | ---- |
|                                   |     |     |     |                  |     |     | sification |     | error | is reduced. |     | We have | demonstrated |     | that |
wellcalibrated,whichisremarkablegiventhatalltheprob-
abilitiesweremodifiedbyonlyasingleparameter. Wein- recent advances in neural network architecture and train-
|     |     |     |     |     |     |     | ing | – model | capacity, |     | normalization, |     | and | regularization |     |
| --- | --- | --- | --- | --- | --- | --- | --- | ------- | --------- | --- | -------------- | --- | --- | -------------- | --- |
cludereliabilitydiagramsforotherdatasetsinSectionS4.
|                                       |               |         |              |             |              |           | –          | have strong |           | effects on  | network   | calibration.  |                | It remains |        |
| ------------------------------------- | ------------- | ------- | ------------ | ----------- | ------------ | --------- | ---------- | ----------- | --------- | ----------- | --------- | ------------- | -------------- | ---------- | ------ |
|                                       |               |         |              |             |              |           | future     | work        | to        | understand  | why       | these         | trends         | affect     | cali-  |
| Computation                           |               | time.   | All methods  | scale       | linearly     | with the  |            |             |           |             |           |               |                |            |        |
|                                       |               |         |              |             |              |           | bration    | while       | improving |             | accuracy. | Nevertheless, |                | simple     |        |
| number                                | of validation |         | set samples. | Temperature |              | scaling   |            |             |           |             |           |               |                |            |        |
|                                       |               |         |              |             |              |           | techniques |             | can       | effectively | remedy    | the           | miscalibration |            | phe-   |
| is by far                             | the           | fastest | method,      | as it       | amounts      | to a one- |            |             |           |             |           |               |                |            |        |
|                                       |               |         |              |             |              |           | nomenon    |             | in neural | networks.   |           | Temperature   |                | scaling    | is the |
| dimensionalconvexoptimizationproblem. |               |         |              |             | Usingaconju- |           |            |             |           |             |           |               |                |            |        |
simplest,fastest,andmoststraightforwardofthemethods,
gategradientsolver,theoptimaltemperaturecanbefound
andsurprisinglyisoftenthemosteffective.
| in10iterations, |     | orafractionofasecondonmostmodern |     |     |     |     |     |     |     |     |     |     |     |     |     |
| --------------- | --- | -------------------------------- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
hardware. Infact,evenanaiveline-searchfortheoptimal 4 http://github.
|     |     |     |     |     |     |     |     | For | an example | implementation, |     | see |     |     |     |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | ---------- | --------------- | --- | --- | --- | --- | --- |
temperature is faster than any of the other methods. The com/gpleiss/temperature_scaling.

Acknowledgments Hannun, Awni, Case, Carl, Casper, Jared, Catanzaro,
|                  |                 |           |        |          |                  |               | Bryan,            | Diamos,  | Greg,                           | Elsen, | Erich,  | Prenger, | Ryan, |
| ---------------- | --------------- | --------- | ------ | -------- | ---------------- | ------------- | ----------------- | -------- | ------------------------------- | ------ | ------- | -------- | ----- |
| The authors      | are             | supported | in     | part by  | the III-1618134, | III-          |                   |          |                                 |        |         |          |       |
|                  |                 |           |        |          |                  |               | Satheesh,         | Sanjeev, | Sengupta,                       |        | Shubho, | Coates,  | Adam, |
| 1526012,         | and IIS-1149882 |           | grants | from     | the              | National Sci- |                   |          |                                 |        |         |          |       |
|                  |                 |           |        |          |                  |               | etal. Deepspeech: |          | Scalingupend-to-endspeechrecog- |        |         |          |       |
| ence Foundation, |                 | as well   | as     | the Bill | and Melinda      | Gates         |                   |          |                                 |        |         |          |       |
nition. arXivpreprintarXiv:1412.5567,2014.
FoundationandtheOfficeofNavalResearch.
|     |     |     |     |     |     |     | He, Kaiming, | Zhang, | Xiangyu, |     | Ren, | Shaoqing, | and Sun, |
| --- | --- | --- | --- | --- | --- | --- | ------------ | ------ | -------- | --- | ---- | --------- | -------- |
References Jian. Deep residual learning for image recognition. In
CVPR,pp.770–778,2016.
| Al-Shedivat, | Maruan,      |     | Wilson,   | Andrew | Gordon,     | Saatchi, |            |     |             |     |        |            |         |
| ------------ | ------------ | --- | --------- | ------ | ----------- | -------- | ---------- | --- | ----------- | --- | ------ | ---------- | ------- |
|              |              |     |           |        |             |          | Hendrycks, | Dan | and Gimpel, |     | Kevin. | A baseline | for de- |
| Yunus,       | Hu, Zhiting, |     | and Xing, | Eric   | P. Learning | scal-    |            |     |             |     |        |            |         |
tectingmisclassifiedandout-of-distributionexamplesin
abledeepkernelswithrecurrentstructure.arXivpreprint
| arXiv:1610.08936,2016. |                    |                          |        |                    |              |          | neuralnetworks.                             |      | InICLR,2017. |              |          |        |             |
| ---------------------- | ------------------ | ------------------------ | ------ | ------------------ | ------------ | -------- | ------------------------------------------- | ---- | ------------ | ------------ | -------- | ------ | ----------- |
|                        |                    |                          |        |                    |              |          | Hinton,Geoffrey,Vinyals,Oriol,andDean,Jeff. |      |              |              |          |        | Distilling  |
| Bengio,                | Yoshua,Goodfellow, |                          |        | IanJ,andCourville, |              | Aaron.   |                                             |      |              |              |          |        |             |
|                        |                    |                          |        |                    |              |          | theknowledgeinaneuralnetwork.               |      |              |              |          | 2015.  |             |
| Deeplearning.          |                    | Nature,521:436–444,2015. |        |                    |              |          |                                             |      |              |              |          |        |             |
|                        |                    |                          |        |                    |              |          | Huang, Gao,                                 | Sun, | Yu,          | Liu, Zhuang, |          | Sedra, | Daniel, and |
| Bojarski,              | Mariusz,           | Del                      | Testa, | Davide,            | Dworakowski, |          |                                             |      |              |              |          |        |             |
|                        |                    |                          |        |                    |              |          | Weinberger,                                 |      | Kilian.      | Deep         | networks | with   | stochastic  |
| Daniel,                | Firner,            | Bernhard,                | Flepp, |                    | Beat, Goyal, | Prasoon, |                                             |      |              |              |          |        |             |
depth. InECCV,2016.
| Jackel, | Lawrence | D,  | Monfort, | Mathew, |     | Muller, Urs, |     |     |     |     |     |     |     |
| ------- | -------- | --- | -------- | ------- | --- | ------------ | --- | --- | --- | --- | --- | --- | --- |
Zhang,Jiakai,etal. Endtoendlearningforself-driving Huang, Gao, Liu, Zhuang, Weinberger, Kilian Q, and
arXivpreprintarXiv:1604.07316,2016.
| cars.    |       |           |         |           |     |             | van der         | Maaten, | Laurens.     | Densely |     | connected | convolu- |
| -------- | ----- | --------- | ------- | --------- | --- | ----------- | --------------- | ------- | ------------ | ------- | --- | --------- | -------- |
|          |       |           |         |           |     |             | tionalnetworks. |         | InCVPR,2017. |         |     |           |          |
| Caruana, | Rich, | Lou, Yin, | Gehrke, | Johannes, |     | Koch, Paul, |                 |         |              |         |     |           |          |
Sturm,Marc,andElhadad,Noemie. Intelligiblemodels Ioffe,SergeyandSzegedy,Christian. Batchnormalization:
for healthcare: Predicting pneumonia risk and hospital Acceleratingdeepnetworktrainingbyreducinginternal
| 30-dayreadmission. |        |              | InKDD,2015. |     |        |              | covariateshift. |             | 2015. |        |              |     |         |
| ------------------ | ------ | ------------ | ----------- | --- | ------ | ------------ | --------------- | ----------- | ----- | ------ | ------------ | --- | ------- |
|                    |        |              |             |     |        |              | Iyyer, Mohit,   | Manjunatha, |       | Varun, | Boyd-Graber, |     | Jordan, |
| Collobert,         | Ronan, | Kavukcuoglu, |             |     | Koray, | and Farabet, |                 |             |       |        |              |     |         |
Cle´ment. Torch7: A matlab-like environment for ma- andDaume´ III,Hal. Deepunorderedcompositionrivals
|                             |     |                               |     |                           |     |     | syntacticmethodsfortextclassification. |                                 |                |     |        | InACL,2015.     |     |
| --------------------------- | --- | ----------------------------- | --- | ------------------------- | --- | --- | -------------------------------------- | ------------------------------- | -------------- | --- | ------ | --------------- | --- |
| chinelearning.              |     | InBigLearnWorkshop,NIPS,2011. |     |                           |     |     |                                        |                                 |                |     |        |                 |     |
|                             |     |                               |     |                           |     |     | Jaynes, Edwin                          |                                 | T. Information |     | theory | and statistical | me- |
| Cosmides,LedaandTooby,John. |     |                               |     | Arehumansgoodintu-        |     |     |                                        |                                 |                |     |        |                 |     |
|                             |     |                               |     |                           |     |     | chanics.                               | Physicalreview,106(4):620,1957. |                |     |        |                 |     |
| itivestatisticiansafterall? |     |                               |     | rethinkingsomeconclusions |     |     |                                        |                                 |                |     |        |                 |     |
fromtheliteratureonjudgmentunderuncertainty. cog- Jiang, Xiaoqian, Osl, Melanie, Kim, Jihoon, and Ohno-
nition,58(1):1–73,1996.
Machado,Lucila.Calibratingpredictivemodelestimates
|                                      |            |     |                 |     |                   |            | tosupportpersonalizedmedicine. |     |             |              | JournaloftheAmer- |                |     |
| ------------------------------------ | ---------- | --- | --------------- | --- | ----------------- | ---------- | ------------------------------ | --- | ----------- | ------------ | ----------------- | -------------- | --- |
| DeGroot,MorrisHandFienberg,StephenE. |            |     |                 |     |                   | Thecompar- |                                |     |             |              |                   |                |     |
|                                      |            |     |                 |     |                   |            | ican Medical                   |     | Informatics | Association, |                   | 19(2):263–274, |     |
| ison and                             | evaluation |     | of forecasters. |     | The statistician, | pp.        |                                |     |             |              |                   |                |     |
2012.
12–22,1983.
Kendall,AlexandCipolla,Roberto.Modellinguncertainty
Deng,Jia,Dong,Wei,Socher,Richard,Li,Li-Jia,Li,Kai,
|              |     |               |     |               |     |              | indeeplearningforcamerarelocalization. |     |     |     |     |     | 2016. |
| ------------ | --- | ------------- | --- | ------------- | --- | ------------ | -------------------------------------- | --- | --- | --- | --- | --- | ----- |
| and Fei-Fei, |     | Li. Imagenet: |     | A large-scale |     | hierarchical |                                        |     |     |     |     |     |       |
imagedatabase. InCVPR,pp.248–255,2009. Kendall, Alex and Gal, Yarin. What uncertainties do we
|                            |     |     |     |                        |     |     | need in | bayesian | deep | learning | for | computer | vision? |
| -------------------------- | --- | --- | --- | ---------------------- | --- | --- | ------- | -------- | ---- | -------- | --- | -------- | ------- |
| Denker,JohnSandLecun,Yann. |     |     |     | Transformingneural-net |     |     |         |          |      |          |     |          |         |
arXivpreprintarXiv:1703.04977,2017.
| output | levels | to probability |     | distributions. |     | In NIPS, pp. |                   |     |        |          |       |      |              |
| ------ | ------ | -------------- | --- | -------------- | --- | ------------ | ----------------- | --- | ------ | -------- | ----- | ---- | ------------ |
|        |        |                |     |                |     |              | Krause, Jonathan, |     | Stark, | Michael, | Deng, | Jia, | and Fei-Fei, |
853–859,1990.
|     |     |     |     |     |     |     | Li. 3d | object | representations |     | for | fine-grained | catego- |
| --- | --- | --- | --- | --- | --- | --- | ------ | ------ | --------------- | --- | --- | ------------ | ------- |
Friedman, Jerome, Hastie, Trevor, andTibshirani, Robert. rization. In IEEE Workshop on 3D Representation and
| Theelementsofstatisticallearning,volume1. |     |     |     |     |     | Springer |     |     |     |     |     |     |     |
| ----------------------------------------- | --- | --- | --- | --- | --- | -------- | --- | --- | --- | --- | --- | --- | --- |
Recognition(3dRR),Sydney,Australia,2013.
seriesinstatisticsSpringer,Berlin,2001.
|     |     |     |     |     |     |     | Krizhevsky,AlexandHinton,Geoffrey. |     |     |     |     | Learningmultiple |     |
| --- | --- | --- | --- | --- | --- | --- | ---------------------------------- | --- | --- | --- | --- | ---------------- | --- |
Gal,YarinandGhahramani,Zoubin.Dropoutasabayesian layersoffeaturesfromtinyimages,2009.
| approximation: |     | Representingmodeluncertaintyindeep |     |     |     |     |     |     |     |     |     |     |     |
| -------------- | --- | ---------------------------------- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
learning. InICML,2016. Kuleshov, Volodymyr and Ermon, Stefano. Reliable con-
|     |     |     |     |     |     |     | fidence | estimation | via | online | learning. | arXiv | preprint |
| --- | --- | --- | --- | --- | --- | --- | ------- | ---------- | --- | ------ | --------- | ----- | -------- |
Girshick,Ross. Fastr-cnn. InICCV,pp.1440–1448,2015. arXiv:1607.03594,2016.

SupplementaryMaterials:OnCalibrationofModernNeuralNetworks
Kuleshov, Volodymyr and Liang, Percy. Calibrated struc- Srivastava, Rupesh Kumar, Greff, Klaus, and Schmid-
turedprediction. InNIPS,pp.3474–3482,2015. huber, Ju¨rgen. Highway networks. arXiv preprint
arXiv:1505.00387,2015.
Lakshminarayanan, Balaji, Pritzel, Alexander, and Blun-
dell, Charles. Simple and scalable predictive uncer- Tai, Kai Sheng, Socher, Richard, and Manning, Christo-
tainty estimation using deep ensembles. arXiv preprint pher D. Improved semantic representations from tree-
arXiv:1612.01474,2016. structuredlongshort-termmemorynetworks. 2015.
Vapnik, Vladimir N. Statistical Learning Theory. Wiley-
LeCun,Yann,Bottou,Le´on,Bengio,Yoshua,andHaffner,
Interscience,1998.
Patrick. Gradient-based learning applied to document
recognition. Proceedings of the IEEE, 86(11):2278–
Welinder, P., Branson, S., Mita, T., Wah, C., Schroff, F.,
2324,1998.
Belongie, S., and Perona, P. Caltech-UCSD Birds 200.
Technical Report CNS-TR-2010-001, California Insti-
MacKay, David JC. A practical bayesian framework for
tuteofTechnology,2010.
backpropagation networks. Neural computation, 4(3):
448–472,1992. Wilson,AndrewG,Hu,Zhiting,Salakhutdinov,RuslanR,
and Xing, Eric P. Stochastic variational deep kernel
Naeini, Mahdi Pakdaman, Cooper, Gregory F, and learning. InNIPS,pp.2586–2594,2016a.
Hauskrecht,Milos. Obtainingwellcalibratedprobabili-
tiesusingbayesianbinning. InAAAI,pp.2901,2015. Wilson,AndrewGordon,Hu,Zhiting,Salakhutdinov,Rus-
lan,andXing,EricP.Deepkernellearning.InAISTATS,
Netzer, Yuval, Wang, Tao, Coates, Adam, Bissacco, pp.370–378,2016b.
Alessandro, Wu, Bo, and Ng, Andrew Y. Reading dig-
Xiong, Wayne, Droppo, Jasha, Huang, Xuedong, Seide,
itsinnaturalimageswithunsupervisedfeaturelearning.
In Deep Learning and Unsupervised Feature Learning Frank, Seltzer, Mike, Stolcke, Andreas, Yu, Dong,
Workshop,NIPS,2011. and Zweig, Geoffrey. Achieving human parity in
conversational speech recognition. arXiv preprint
Niculescu-Mizil,AlexandruandCaruana,Rich. Predicting arXiv:1610.05256,2016.
good probabilities with supervised learning. In ICML,
Zadrozny, Bianca and Elkan, Charles. Obtaining cal-
pp.625–632,2005.
ibrated probability estimates from decision trees and
naivebayesianclassifiers. InICML,pp.609–616,2001.
Pereyra,Gabriel,Tucker,George,Chorowski,Jan,Kaiser,
Łukasz, and Hinton, Geoffrey. Regularizing neural
Zadrozny,BiancaandElkan,Charles.Transformingclassi-
networks by penalizing confident output distributions.
fierscoresintoaccuratemulticlassprobabilityestimates.
arXivpreprintarXiv:1701.06548,2017.
InKDD,pp.694–699,2002.
Platt, John et al. Probabilistic outputs for support vec- Zagoruyko,SergeyandKomodakis,Nikos. Wideresidual
tormachinesandcomparisonstoregularizedlikelihood networks. InBMVC,2016.
methods. Advances in large margin classifiers, 10(3):
61–74,1999. Zhang,Chiyuan,Bengio,Samy,Hardt,Moritz,Recht,Ben-
jamin,andVinyals,Oriol. Understandingdeeplearning
Simonyan,KarenandZisserman,Andrew. Verydeepcon- requiresrethinkinggeneralization. InICLR,2017.
volutionalnetworksforlarge-scaleimagerecognition.In
ICLR,2015.
Socher, Richard, Perelygin, Alex, Wu, Jean, Chuang, Ja-
son, Manning, Christopher D., Ng, Andrew, and Potts,
Christopher. Recursive deep models for semantic com-
positionalityoverasentimenttreebank. InEMNLP,pp.
1631–1642,2013.
Srivastava, Nitish, Hinton, Geoffrey, Krizhevsky, Alex,
Sutskever,Ilya,andSalakhutdinov,Ruslan. Dropout: A
simplewaytopreventneuralnetworksfromoverfitting.
Journal of Machine Learning Research, 15:1929–1958,
2014.

Supplementary Materials for:
On Calibration of Modern Neural Networks
S1.FurtherInformationonCalibration The first two constraint ensure that q is a probability dis-
Metrics tribution,whilethelastconstraintlimitsthescopeofdistri-
butions.Intuitively,theconstraintspecifiesthattheaverage
WecanconnecttheECEmetricwithourexactmiscalibra- trueclasslogitisequaltotheaverageweightedlogit.
tiondefinition,whichisrestatedhere:
(cid:104)(cid:12) (cid:16) (cid:17) (cid:12)(cid:105)
E (cid:12)P Yˆ =Y |Pˆ =p −p(cid:12) Proof. Wesolvethisconstrainedoptimizationproblemus-
(cid:12) (cid:12)
Pˆ ingtheLagrangian. Wefirstignoretheconstraintq(z )(k)
i
LetF (p)bethecumulativedistributionfunctionofPˆ so andlatershowthatthesolutionsatisfiesthiscondition. Let
Pˆ
thatF Pˆ (b)−F Pˆ (a)=P(Pˆ ∈[a,b]). UsingtheRiemann- λ,β 1 ,...,β n ∈RbetheLagrangianmultipliersanddefine
Stieltjesintegralwehave n K
(cid:88)(cid:88)
(cid:104)(cid:12) (cid:16) (cid:17) (cid:12)(cid:105) L=− q(z )(k)logq(z )(k)
E (cid:12)P Yˆ =Y |Pˆ =p −p(cid:12) i i
(cid:12) (cid:12)
Pˆ i=1k=1
= (cid:90) 1(cid:12) (cid:12) (cid:12) P (cid:16) Yˆ =Y |Pˆ =p (cid:17) −p (cid:12) (cid:12) (cid:12) dF Pˆ (p) +λ (cid:88) n (cid:34) (cid:88) K z i (k)q(z i )(k)−z i (yi) (cid:35)
0 i=1 k=1
≈ (cid:88) M (cid:12) (cid:12) (cid:12) P(Yˆ =Y|Pˆ =p m )−p m (cid:12) (cid:12) (cid:12) P(Pˆ ∈I m ) + (cid:88) n β i (cid:88) K (q(z i )(k)−1).
m=1
i=1 k=1
where I represents the interval of bin B .
(cid:12) (cid:12)P(Yˆ =Y m |Pˆ =p )−p (cid:12) (cid:12) is closely approximat m ed Takingthederivativewithrespecttoq(z i )(k)gives
(cid:12) m m(cid:12) ∂
L=−nK−logq(z )(k)+λz(k)+β .
by |acc(B m )−pˆ(B m )| for n large. Hence ECE using M ∂q(z )(k) i i i
i
bins converges to the M-term Riemann-Stieltjes sum of
(cid:104)(cid:12) (cid:16) (cid:17) (cid:12)(cid:105) SettingthegradientoftheLagrangianLto0andrearrang-
E (cid:12)P Yˆ =Y |Pˆ =p −p(cid:12) .
Pˆ (cid:12) (cid:12) inggives
q(z i )(k) =eλz i (k)+βi−nK.
S2.FurtherInformationonTemperature
Scaling Since (cid:80)K k=1 q(z i )(k) =1foralli,wemusthave
Herewederivethetemperaturescalingmodelusingtheen- q(z )(k) =
eλz
i
(k)
,
tropymaximizationprinciplewithanappropriatebalanced i (cid:80)K
j=1
eλz
i
(j)
equation.
which recovers the temperature scaling model by setting
Claim 1. Given n samples’ logit vectors z 1 ,...,z n and T = 1.
λ
class labels y ,...,y , temperature scaling is the unique
1 n
solutionqtothefollowingentropymaximizationproblem:
FigureS1visualizesClaim1. Weseethat,astrainingcon-
(cid:88) n (cid:88) K tinues,themodelbeginstooverfitwithrespecttoNLL(red
max − q(z )(k)logq(z )(k)
i i line). This results in a low-entropy softmax distribution
q
i=1k=1 over classes (blue line), which explains the model’s over-
subjectto q(z i )(k) ≥0 ∀i,k confidence. TemperaturescalingnotonlylowerstheNLL
K butalsoraisestheentropyofthedistribution(greenline).
(cid:88)
q(z )(k) =1 ∀i
i
k=1 S3.AdditionalTables
n n K
(cid:88) z i (yi) = (cid:88)(cid:88) z i (k)q(z i )(k). TablesS1,S2,andS3displaytheMCE,testerror,andNLL
i=1 i=1k=1 foralltheexperimentalsettingsoutlinedinSection5.

SupplementaryMaterials:OnCalibrationofModernNeuralNetworks
3.5
3
2.5
2
1.5
1
0.5
0 100 200 300 400 500
Epoch
T
/
LLN
/
yportnE
Entropy vs. NLL on CIFAR−100
Entropy & NLL after Calibration
Entropy before Calibration
NLL before Calibration
Optimal T Selected
FigureS1.EntropyandNLLforCIFAR-100beforeandaftercalibration.TheoptimalT selectedbytemperaturescalingrisesthroughout
optimization, as the pre-calibration entropy decreases steadily. The post-calibration entropy and NLL on the validation set coincide
(whichcanbederivedfromthegradientoptimalityconditionofT).
Dataset Model Uncalibrated Hist.Binning Isotonic BBQ Temp.Scaling VectorScaling MatrixScaling
Birds ResNet50 30.06% 25.35% 16.59% 11.72% 9.08% 9.81% 38.67%
Cars ResNet50 41.55% 5.16% 15.23% 9.31% 20.23% 8.59% 29.65%
CIFAR-10 ResNet110 33.78% 26.87% 7.8% 72.64% 8.56% 27.39% 22.89%
CIFAR-10 ResNet110(SD) 34.52% 17.0% 16.45% 19.26% 15.45% 15.55% 10.74%
CIFAR-10 WideResNet32 27.97% 12.19% 6.19% 9.22% 9.11% 4.43% 9.65%
CIFAR-10 DenseNet40 22.44% 7.77% 19.54% 14.57% 4.58% 3.17% 4.36%
CIFAR-10 LeNet5 8.02% 16.49% 18.34% 82.35% 5.14% 19.39% 16.89%
CIFAR-100 ResNet110 35.5% 7.03% 10.36% 10.9% 4.74% 2.5% 45.62%
CIFAR-100 ResNet110(SD) 26.42% 9.12% 10.95% 9.12% 8.85% 8.85% 35.6%
CIFAR-100 WideResNet32 33.11% 6.22% 14.87% 11.88% 5.33% 6.31% 44.73%
CIFAR-100 DenseNet40 21.52% 9.36% 10.59% 8.67% 19.4% 8.82% 38.64%
CIFAR-100 LeNet5 10.25% 18.61% 3.64% 9.96% 5.22% 8.65% 18.77%
ImageNet DenseNet161 14.07% 13.14% 11.57% 10.96% 12.29% 9.61% -
ImageNet ResNet152 12.2% 14.57% 8.74% 8.85% 12.29% 9.61% -
SVHN ResNet152(SD) 19.36% 11.16% 18.67% 9.09% 18.05% 30.78% 18.76%
20News DAN3 17.03% 10.47% 9.13% 6.28% 8.21% 8.24% 17.43%
Reuters DAN3 14.01% 16.78% 44.95% 36.18% 25.46% 18.88% 19.39%
SSTBinary TreeLSTM 21.66% 3.22% 13.91% 36.43% 6.03% 6.03% 6.03%
SSTFineGrained TreeLSTM 27.85% 28.35% 19.0% 8.67% 44.75% 11.47% 11.78%
TableS1.MCE(%)(withM =15bins)onstandardvisionandNLPdatasetsbeforecalibrationandwithvariouscalibrationmethods.
Thenumberfollowingamodel’snamedenotesthenetworkdepth. MCEseemsverysensitivetothebinningschemeandislesssuited
forsmalltestsets.
S4.AdditionalReliabilityDiagrams grams do not represent the proportion of predictions that
belongtoagivenbin.
We include reliability diagrams for additional datasets:
CIFAR-10(FigureS2)andSST(FigureS3andFigureS4).
Note that, as mentioned in Section 2, the reliability dia-

SupplementaryMaterials:OnCalibrationofModernNeuralNetworks
Dataset Model Uncalibrated Hist.Binning Isotonic BBQ Temp.Scaling VectorScaling MatrixScaling
Birds ResNet50 22.54% 55.02% 23.37% 37.76% 22.54% 22.99% 29.51%
Cars ResNet50 14.28% 16.24% 14.9% 19.25% 14.28% 14.15% 17.98%
CIFAR-10 ResNet110 6.21% 6.45% 6.36% 6.25% 6.21% 6.37% 6.42%
CIFAR-10 ResNet110(SD) 5.64% 5.59% 5.62% 5.55% 5.64% 5.62% 5.69%
CIFAR-10 WideResNet32 6.96% 7.3% 7.01% 7.35% 6.96% 7.1% 7.27%
CIFAR-10 DenseNet40 5.91% 6.12% 5.96% 6.0% 5.91% 5.96% 6.0%
CIFAR-10 LeNet5 15.57% 15.63% 15.69% 15.64% 15.57% 15.53% 15.81%
CIFAR-100 ResNet110 27.83% 34.78% 28.41% 28.56% 27.83% 27.82% 38.77%
CIFAR-100 ResNet110(SD) 24.91% 33.78% 25.42% 25.17% 24.91% 24.99% 35.09%
CIFAR-100 WideResNet32 28.0% 34.29% 28.61% 29.08% 28.0% 28.45% 37.4%
CIFAR-100 DenseNet40 26.45% 34.78% 26.73% 26.4% 26.45% 26.25% 36.14%
CIFAR-100 LeNet5 44.92% 54.06% 45.77% 46.82% 44.92% 45.53% 52.44%
ImageNet DenseNet161 22.57% 48.32% 23.2% 47.58% 22.57% 22.54% -
ImageNet ResNet152 22.31% 48.1% 22.94% 47.6% 22.31% 22.56% -
SVHN ResNet152(SD) 1.98% 2.06% 2.04% 2.04% 1.98% 2.0% 2.08%
20News DAN3 20.06% 25.12% 20.29% 20.81% 20.06% 19.89% 22.0%
Reuters DAN3 2.97% 7.81% 3.52% 3.93% 2.97% 2.83% 3.52%
SSTBinary TreeLSTM 11.81% 12.08% 11.75% 11.26% 11.81% 11.81% 11.81%
SSTFineGrained TreeLSTM 49.5% 49.91% 48.55% 49.86% 49.5% 49.77% 48.51%
TableS2.Test error (%) on standard vision and NLP datasets before calibration and with various calibration methods. The number
followingamodel’snamedenotesthenetworkdepth.Errorwithtemperaturescalingisexactlythesameasuncalibrated.
Dataset Model Uncalibrated Hist.Binning Isotonic BBQ Temp.Scaling VectorScaling MatrixScaling
Birds ResNet50 0.9786 1.6226 1.4128 1.2539 0.8792 0.9021 2.334
Cars ResNet50 0.5488 0.7977 0.8793 0.6986 0.5311 0.5299 1.0206
CIFAR-10 ResNet110 0.3285 0.2532 0.2237 0.263 0.2102 0.2088 0.2048
CIFAR-10 ResNet110(SD) 0.2959 0.2027 0.1867 0.2159 0.1718 0.1709 0.1766
CIFAR-10 WideResNet32 0.3293 0.2778 0.2428 0.2774 0.2283 0.2275 0.2229
CIFAR-10 DenseNet40 0.2228 0.212 0.1969 0.2087 0.1750 0.1757 0.176
CIFAR-10 LeNet5 0.4688 0.529 0.4757 0.4984 0.459 0.4568 0.4607
CIFAR-100 ResNet110 1.4978 1.4379 1.207 1.5466 1.0442 1.0485 2.5637
CIFAR-100 ResNet110(SD) 1.1157 1.1985 1.0317 1.1982 0.8613 0.8655 1.8182
CIFAR-100 WideResNet32 1.3434 1.4499 1.2086 1.459 1.0565 1.0648 2.5507
CIFAR-100 DenseNet40 1.0134 1.2156 1.0615 1.1572 0.9026 0.9011 1.9639
CIFAR-100 LeNet5 1.6639 2.2574 1.8173 1.9893 1.6560 1.6648 2.1405
ImageNet DenseNet161 0.9338 1.4716 1.1912 1.4272 0.8885 0.8879 -
ImageNet ResNet152 0.8961 1.4507 1.1859 1.3987 0.8657 0.8742 -
SVHN ResNet152(SD) 0.0842 0.1137 0.095 0.1062 0.0821 0.0844 0.0924
20News DAN3 0.7949 1.0499 0.8968 0.9519 0.7387 0.7296 0.9089
Reuters DAN3 0.102 0.2403 0.1475 0.1167 0.0994 0.0990 0.1491
SSTBinary TreeLSTM 0.3367 0.2842 0.2908 0.2778 0.2739 0.2739 0.2739
SSTFineGrained TreeLSTM 1.1475 1.1717 1.1661 1.149 1.1168 1.1085 1.1112
TableS3.NLL(%)onstandardvisionandNLPdatasetsbeforecalibrationandwithvariouscalibrationmethods.Thenumberfollowing
amodel’snamedenotesthenetworkdepth.Tosummarize,NLLroughlyfollowsthetrendsofECE.

SupplementaryMaterials:OnCalibrationofModernNeuralNetworks
Uncal. -CIFAR-10 Temp. Scale-CIFAR-10 Hist. Bin. -CIFAR-10 Iso. Reg. -CIFAR-10
|     | ResNet-110(SD) | ResNet-110(SD) | ResNet-110(SD) | ResNet-110(SD) |
| --- | -------------- | -------------- | -------------- | -------------- |
1.0
|     | Outputs | Outputs | Outputs | Outputs |
| --- | ------- | ------- | ------- | ------- |
| 0.8 | Gap     | Gap     | Gap     | Gap     |
ycaruccA 0.6
0.4
0.2
|     | ECE=4.12 | ECE=0.60 | ECE=0.67 | ECE=1.11 |
| --- | -------- | -------- | -------- | -------- |
0.0
0.0 0.2 0.4 0.6 0.8 1.0 0.0 0.2 0.4 0.6 0.8 1.0 0.0 0.2 0.4 0.6 0.8 1.0 0.0 0.2 0.4 0.6 0.8 1.0
Confidence
FigureS2. ReliabilitydiagramsforCIFAR-10before(farleft)andaftercalibration(middleleft,middleright,farright).
Uncal. -SST-FG Temp. Scale-SST-FG Hist. Bin. -SST-FG Iso. Reg. -SST-FG
|     | TreeLSTM | TreeLSTM | TreeLSTM | TreeLSTM |
| --- | -------- | -------- | -------- | -------- |
1.0
|     | Outputs | Outputs | Outputs | Outputs |
| --- | ------- | ------- | ------- | ------- |
0.8
|     | Gap | Gap | Gap | Gap |
| --- | --- | --- | --- | --- |
ycaruccA
0.6
0.4
0.2
|     | ECE=6.71 | ECE=2.56 | ECE=2.09 | ECE=1.65 |
| --- | -------- | -------- | -------- | -------- |
0.0
0.0 0.2 0.4 0.6 0.8 1.0 0.0 0.2 0.4 0.6 0.8 1.0 0.0 0.2 0.4 0.6 0.8 1.0 0.0 0.2 0.4 0.6 0.8 1.0
Confidence
FigureS3.ReliabilitydiagramsforSSTBinaryandSSTFineGrainedbefore(farleft)andaftercalibration(middleleft,middleright,
farright).
Uncal. -SST-BIN Temp. Scale-SST-BIN Hist. Bin. -SST-BIN Iso. Reg. -SST-BIN
|     | TreeLSTM | TreeLSTM | TreeLSTM | TreeLSTM |
| --- | -------- | -------- | -------- | -------- |
1.0
|     | Outputs | Outputs | Outputs | Outputs |
| --- | ------- | ------- | ------- | ------- |
| 0.8 | Gap     | Gap     | Gap     | Gap     |
ycaruccA
0.6
0.4
0.2
|     | ECE=6.63 | ECE=1.84 | ECE=1.93 | ECE=1.65 |
| --- | -------- | -------- | -------- | -------- |
0.0
0.0 0.2 0.4 0.6 0.8 1.0 0.0 0.2 0.4 0.6 0.8 1.0 0.0 0.2 0.4 0.6 0.8 1.0 0.0 0.2 0.4 0.6 0.8 1.0
Confidence
FigureS4.ReliabilitydiagramsforSSTBinaryandSSTFineGrainedbefore(farleft)andaftercalibration(middleleft,middleright,
farright).
