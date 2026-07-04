PublishedasaconferencepaperatICLR2018
ALL-BUT-THE-TOP: SIMPLE AND EFFECTIVE POST-
PROCESSING FOR WORD REPRESENTATIONS
JiaqiMu,PramodViswanath
UniversityofIllinoisatUrbanaChampaign
{jiaqimu2, pramodv}@illinois.edu
ABSTRACT
Real-valued word representations have transformed NLP applications; popular
examplesareword2vecandGloVe,recognizedfortheirabilitytocapturelinguistic
regularities. Inthispaper,wedemonstrateaverysimple,andyetcounter-intuitive,
postprocessing technique – eliminate the common mean vector and a few top
dominatingdirectionsfromthewordvectors–thatrendersoff-the-shelfrepresen-
tationsevenstronger. Thepostprocessingisempiricallyvalidatedonavarietyof
lexical-levelintrinsictasks(wordsimilarity,conceptcategorization,wordanalogy)
andsentence-leveltasks(semantictexturalsimilarityand textclassification)on
multipledatasetsandwithavarietyofrepresentationmethodsandhyperparameter
choices in multiple languages; in each case, the processed representations are
consistentlybetterthantheoriginalones.
1 INTRODUCTION
Wordsandtheirinteractions(assentences)arethebasicunitsofnaturallanguage. Althoughwords
are readily modeled as discrete atomic units, this is unable to capture the relation between the
words. Recentdistributionalreal-valuedrepresentationsofwords(examples: word2vec,GloVe)have
transformedthelandscapeofNLPapplications–forinstance,textclassification(Socheretal.,2013b;
Maasetal.,2011;Kim,2014),machinetranslation(Sutskeveretal.,2014;Bahdanauetal.,2014)and
knowledgebasecompletion(Bordesetal.,2013;Socheretal.,2013a). Thesuccesscomesfromthe
geometryoftherepresentationsthatefficientlycaptureslinguisticregularities: thesemanticsimilarity
ofwordsiswellcapturedbythesimilarityofthecorrespondingvectorrepresentations.
A variety of approaches have been proposed in recent years to learn the word representations:
Collobertetal.(2011);Turianetal.(2010)learntherepresentationsviasemi-supervisedlearningby
jointlytrainingthelanguagemodelanddownstreamapplications;Bengioetal.(2003);Mikolovetal.
(2010);Huangetal.(2012)dosobyfittingthedataintoaneuralnetworklanguagemodel;Mikolov
etal.(2013);Mnih&Hinton(2007)bylog-linearmodels;andDhillonetal.(2012);Pennington
etal.(2014);Levy&Goldberg(2014);Stratosetal.(2015);Aroraetal.(2016)byproducingalow-
dimensionalrepresentationofthecooccurrencestatistics. Despitethewidedisparityofalgorithmsto
inducewordrepresentations,theperformanceofseveraloftherecentmethodsisroughlysimilarona
varietyofintrinsicandextrinsicevaluationtestbeds.
Inthispaper,wefindthatasimpleprocessingrenderstheoff-the-shelfexistingrepresentationseven
stronger. Theproposedalgorithmismotivatedbythefollowingobservation.
Observation Everyrepresentationwetested,inmanylanguages,hasthefollowingproperties:
• Thewordrepresentationshavenon-zeromean–indeed,wordvectorssharealargecommon
vector(withnormuptoahalfoftheaveragenormofwordvector).
• Afterremovingthecommonmeanvector,therepresentationsarefarfromisotropic–indeed,
muchoftheenergyofmostwordvectorsiscontainedinaverylowdimensionalsubspace
(say,8dimensionsoutof300).
Implication Sinceallwordssharethesamecommonvectorandhavethesamedominatingdirec-
tions,andsuchvectoranddirectionsstronglyinfluencethewordrepresentationsinthesameway,we
1
8102
raM
91
]LC.sc[
2v71410.2071:viXra

PublishedasaconferencepaperatICLR2018
proposetoeliminatethemby: (a)removingthenonzeromeanvectorfromallwordvectors,effec-
tivelyreducingtheenergy;(b)projectingtherepresentationsawayfromthedominatingDdirections,
effectivelyreducingthedimension. ExperimentssuggestthatDdependsontherepresentations(for
example,thedimensionoftherepresentation,thetrainingmethodsandtheirspecifichyperparameters,
thetrainingcorpus)andalsodependsonthedownstreamapplications. Nevertheless,aruleofthumb
ofchoosingDaroundd/100,wheredisthedimensionofthewordrepresentations,worksuniformly
wellacrossmultiplelanguagesandmultiplerepresentationsandmultipletestscenarios.
Weemphasizethattheproposedpostprocessingiscounterintuitive–typicallydenoisingbydimen-
sionalityreductionisdonebyeliminatingtheweakestdirections(inasingularvaluedecomposition
ofthestackedwordvectors),andnotthedominatingones.Yet,suchpostprocessingyieldsa“purified”
andmore“isotropic”wordrepresentationasseeninourelaborateexperiments.
Experiments By postprocessingthe word representation by eliminating thecommon parts, we
findtheprocessedwordrepresentationstocapturestrongerlinguisticregularities. Wedemonstrate
thisquantitatively,bycomparingtheperformanceofboththeoriginalwordrepresentationsandthe
processedonesonthreecanonicallexical-leveltasks:
• word similarity task tests the extent to which the representations capture the similarity
betweentwowords–theprocessedrepresentationsareconsistentlybetteronsevendifferent
datasets,onaverageby1.7%;
• conceptcategorizationtaskteststheextenttowhichtheclustersofwordrepresentations
capturethewordsemantics–theprocessedrepresentationsareconsistentlybetteronthree
differentdatasets,by2.8%,4.5%and4.3%;
• wordanalogytaskteststheextenttowhichthedifferenceoftworepresentationscaptures
a latent linguistic relation – again, the performance is consistently improved (by 0.5%
onsemanticanalogies,0.2%onsyntacticanalogiesand0.4%intotal). Sincepartofthe
dominantcomponentsareinherentlycanceledduetothesubtractionoperationwhilesolving
theanalogy,wepositthattheperformanceimprovementisnotaspronouncedasearlier.
Extrinsicevaluationsprovideawaytotestthegoodnessofrepresentationsinspecificdownstream
tasks. Weevaluatetheeffectofpostprocessingonastandardizedandimportantextrinsicevaluation
taskonsentencemodeling: semantictextualsimilaritytask–wherewerepresentasentencebyits
averagedwordvectorsandscorethesimilaritybetweenapairofsentencesbythecosinesimilarity
between the corresponding sentence representation. Postprocessing improves the performance
consistentlyandsignificantlyover21differentdatasets(averageimprovementof4%).
WordrepresentationshavebeenparticularlysuccessfulinNLPapplicationsinvolvingsupervised-
learning, especially in conjunction with neural network architecture. Indeed, we see the power
ofpostprocessinginanexperimentonastandardtextclassificationtaskusingawellestablished
convoluntionalneuralnetwork(CNN)classifier(Kim,2014)andthreeRNNclassifiers(withvanilla
RNN, GRU (Chung et al., 2015) and LSTM Greff et al. (2016) as recurrent units). Across two
differentpre-trainedwordvectors,fivedatasetsandfourdifferentarchitectures,theperformancewith
processingimprovesonamajorityofinstances(34outof40)byagoodmargin(2.85%onaverage),
andthetwoperformanceswithandwithoutprocessingarecomparableintheremainingones.
RelatedWork. Ourworkisdirectlyrelatedtowordrepresentationalgorithms,mostofwhichhave
beenelaboratelycited.
AspectssimilartoourpostprocessingalgorithmhaveappearedinspecificNLPcontextsveryrecently
in(Sahlgrenetal.,2016)(centeringthemean)and(Aroraetal.,2017)(nullingawayonlythefirst
principalcomponent). Althoughthereisasuperficialsimilaritybetweenourworkand(Aroraetal.
2017),thenullingdirectionswetakeandtheonetheytakearefundamentallydifferent. Specifically,
inAroraetal. (2017),thefirstdominatingvectoris*dataset-specific*,i.e.,theyfirstcomputethe
sentencerepresentationfortheentiresemantictextualsimilaritydataset,thenextractthetopdirection
from those sentence representations and finally project the sentence representation away from it.
By doing so, the top direction will inherently encode the common information across the entire
dataset,thetopdirectionforthe"headlines"datasetmayencodecommoninformationaboutnews
articleswhilethetopdirectionfor"Twitter’15"mayencodethecommoninformationabouttweets.
Incontrast,ourdominatingvectorsareovertheentirevocabularyofthelanguage.
2

PublishedasaconferencepaperatICLR2018
Moregenerally,theideaofremovingthetopprincipalcomponentshasbeenstudiedinthecontextof
positive-valued,high-dimensionaldatamatrixanalysis(Bullinaria&Levy,2012;Priceetal.,2006).
Bullinaria&Levy(2012)positsthatthehighestvariancecomponentsofthecooccurrencematrixare
corruptedbyinformationotherthanlexicalsemantics,thusheuristicallyjustifyingtheremovalof
thetopprincipalcomponents. Asimilarideaappearsinthecontextofpopulationmatrixanalysis
(Priceetal.,2006),wheretheentriesarealsoallpositive. Ourpostprocessingoperationisondense
low-dimensionalrepresentations(withbothpositiveandnegativeentries).
Wepositthatthepostprocessingoperationmakestherepresentationsmore“isotropic”withstronger
self-normalizationproperties–discussedindetailinSection 2andAppendixA.Ourmainpointis
thatthisisotropyconditioncanbeexplicitlyenforcedtocomeupwithnewembeddingalgorithms(of
whichourproposedpost-processingisasimpleandpracticalversion).
2 POSTPROCESSING
Wetestourobservationsonvariouswordrepresentations:fourpubliclyavailablewordrepresentations
(WORD2VEC1(Mikolovetal.,2013)trainedusingGoogleNews,GLOVE2(Penningtonetal.,2014)
trained using Common Crawl, RAND-WALK (Arora et al., 2016) trained using Wikipedia and
TSCCA3trainedusingEnglishGigaword)andtwoself-trainedwordrepresentationsusingCBOW
andSkip-gram(Mikolovetal.,2013)onthe2010Wikipediacorpusfrom(Al-Rfouetal.,2013). The
detailedstatisticsforallrepresentationsarelistedinTable1. Forcompleteness,wealsoconsiderthe
| representationsonotherlanguages: |          | adetailedstudyisprovidedinAppendixC.2. |               |      |                                            |
| -------------------------------- | -------- | -------------------------------------- | ------------- | ---- | ------------------------------------------ |
|                                  | Language | Corpus                                 | dim vocabsize | avg. | (cid:107)v(w)(cid:107) (cid:107)µ(cid:107) |
2 2
| WORD2VEC  | English | GoogleNews  | 300 3,000,000 | 2.04 | 0.69 |
| --------- | ------- | ----------- | ------------- | ---- | ---- |
| GLOVE     | English | CommonCrawl | 300 2,196,017 | 8.30 | 3.15 |
| RAND-WALK | English | Wikipedia   | 300 68,430    | 2.27 | 0.70 |
| CBOW      | English | Wikipedia   | 300 1,028,961 | 1.14 | 0.29 |
| Skip-Gram | English | Wikipedia   | 300 1,028,961 | 2.32 | 1.25 |
Table1: Adetaileddescriptionfortheembeddingsinthispaper.
Letv(w)∈Rdbeawordrepresentationforagivenwordwinthe
|     |     | vocabularyV. | Weobservethefollowingtwophenomenaineachof |     |     |
| --- | --- | ------------ | ----------------------------------------- | --- | --- |
0.045
GLOVE
| 0.040 | RAND_WALK | thewordrepresentationslistedabove: |     |     |     |
| ----- | --------- | ---------------------------------- | --- | --- | --- |
0.035
| oitar ecnairav | WORD2VEC |     |     |     |     |
| -------------- | -------- | --- | --- | --- | --- |
0.030 CBOW • {v(w) : w ∈ V} are not of zero-mean: i.e., all v(w) share a
| 0.025 | SKIP-GRAM |          |                     |         |                     |
| ----- | --------- | -------- | ------------------- | ------- | ------------------- |
|       |           | non-zero | common vector, v(w) | = v˜(w) | + µ, where µ is the |
0.020
0.015 averageofallv(w)’s,i.e.,µ=1/|V| (cid:80) v(w). Thenormof
w∈V
0.010
µisapproximately1/6to1/2oftheaveragenormofallv(w)(cf.
0.005
| 0.000 |         | Table1). |     |     |     |
| ----- | ------- | -------- | --- | --- | --- |
| 100   | 101 102 | 103      |     |     |     |
index
|     |     | • {v˜(w):w | ∈V}arenotisotropic: | Letu ,...,u | bethefirsttothe |
| --- | --- | ---------- | ------------------- | ----------- | --------------- |
|     |     |            |                     | 1           | d               |
lastcomponentsrecoveredbytheprincipalcomponentanalysis
|               |              | (PCA)of{v˜(w) | : w ∈ V},andσ | ,...,σ bethecorresponding |     |
| ------------- | ------------ | ------------- | ------------- | ------------------------- | --- |
| Figure 1: The | decay of the |               |               | 1 d                       |     |
Eachv˜(w)canbewrittenasalinear
| normalizedsingularvaluesof |     | normalizedvarianceratio. |     |     |     |
| -------------------------- | --- | ------------------------ | --- | --- | --- |
(cid:80)d
wordrepresentation. combinationsofu:v˜(w)= α (w)u .AsshowninFigure1,
i=1 i i
|     |     | weobservethatσ | decaysnearexponentiallyforsmallvaluesof |     |     |
| --- | --- | -------------- | --------------------------------------- | --- | --- |
i
|     |     | iandremainsroughlyconstantoverthelaterones. |     |     | Thissuggests |
| --- | --- | ------------------------------------------- | --- | --- | ------------ |
thereexistsDsuchthatα (cid:29)α foralli≤Dandj (cid:29)D;fromFigure1oneobservesthatDis
i j
roughly10withdimensiond=300.
Angular Asymmetry of Representations A modern understanding of word representations in-
volveseitherPMI-based(includingword2vec(Mikolovetal.,2010;Levy&Goldberg,2014)and
GloVe(Penningtonetal.,2014))orCCA-basedspectralfactorizationapproaches. WhileCCA-based
1https://code.google.com/archive/p/word2vec/
2https://github.com/stanfordnlp/GloVe
3http://www.pdhillon.com/code.html
3

PublishedasaconferencepaperatICLR2018
spectralfactorizationmethodshavelongbeenunderstoodfromaprobabilistic(i.e.,generativemodel)
viewpoint(Browne,1979;Hotelling,1936)andrecentlyintheNLPcontext(Stratosetal.,2015),
acorrespondingeffortforthePMI-basedmethodshasonlyrecentlybeenconductedinaninspired
work(Aroraetal.,2016).
Aroraetal.(2016)proposeagenerativemodel(namedRAND-WALK)ofsentences,whereevery
wordisparameterizedbyad-dimensionalvector. Withakeypostulatethatthewordvectorsare
angularly uniform (“isotropic"), the family of PMI-based word representations can be explained
undertheRAND-WALKmodelintermsofthemaximumlikelihoodrule. Ourobservationthatword
vectorslearntthroughPMI-basedapproachesarenotofzero-meanandarenotisotropic(c.f.Section
2)contradictswiththispostulate. TheisotropyconditionsarerelaxedinSection2.2of(Aroraetal.,
2016),butthematchwiththespectralpropertiesobservedinFigure1isnotimmediate.
Thiscontradictionisexplicitlyreslovedbyrelaxingtheconstraintsonthewordvectorstodirectlyfit
theobservedspectralproperties. Therelaxedconditionsare: thewordvectorsshouldbeisotropic
aroundapoint(whosedistancetotheoriginisasmallfractionoftheaveragenormofwordvectors)
lying on a low dimensional subspace. Our main result is to show that even with this enlarged
parameter-space, the maximum likelihood rule continues to be close to the PMI-based spectral
factorizationmethods. AbriefsummaryofRAND-WALK,andthemathematicalconnectionbetween
ourworkandtheirs,areexploredindetailinAppendixA.
2.1 ALGORITHM
µ
Since all word representations share the same common vector and have the same dominating
directionsandsuchvectoranddirectionsstronglyinfluencethewordrepresentationsinthesameway,
weproposetoeliminatethem,asformallyachievedasAlgorithm1.
Algorithm1:Postprocessingalgorithmonwordrepresentations.
| Input :Wordrepresentations{v(w),w |                     | ∈V},athresholdparameterD,    |         |
| --------------------------------- | ------------------- | ---------------------------- | ------- |
| Computethemeanof{v(w),w           | ∈V},µ←              | 1 (cid:80) v(w),v˜(w)←v(w)−µ |         |
| 1                                 |                     | |V| w∈V                      |         |
| ComputethePCAcomponents:          | u ,...,u            | ←PCA({v˜(w),w                | ∈V}).   |
| 2                                 | 1                   | d                            |         |
|                                   |                     | (cid:80)D (cid:0)            | (cid:1) |
| 3 Preprocesstherepresentations:   | v(cid:48)(w)←v˜(w)− | u(cid:62)v(w)                | u       |
|                                   |                     | i=1                          | i i     |
Output:Processedrepresentationsv(cid:48)(w).
SignificanceofNulledVectors Considertherepresentationofthewordsasviewedintermsof
the top D PCA coefficients α (w), for 1 ≤ (cid:96) ≤ D. We find that these few coefficients encode
(cid:96)
the frequency of the word to a significant degree; Figure 2 illustrates the relation between the
(α (w),α (w))andtheunigramprobabiltyp(w),wherethecorrelationisgeometricallyvisible.
1 2
| Figure2: ThetoptwoPCAdirections(i.e,α |     | (w)andα | (w))encodefrequency. |
| ------------------------------------- | --- | ------- | -------------------- |
|                                       |     | 1       | 2                    |
Discussion Inourproposedprocessingalgorithm,thenumberofcomponentstobenulled,D,is
theonlyhyperparameterthatneedstobetuned. WefindthatagoodruleofthumbistochooseD
approximatelytobed/100,wheredisthedimensionofawordrepresentation. Thisisempirically
justifiedintheexperimentsofthefollowingsectionwhered=300isstandardforpublishedword
representations. WetrainedwordrepresentationsforhighervaluesofdusingtheWORD2VECand
4

PublishedasaconferencepaperatICLR2018
GLOVEalgorithmsandrepeatedtheseexperiments;weseecorrespondingconsistentimprovements
duetopostprocessinginAppendixC.
2.2 POSTPROCESSINGASA“ROUNDING”TOWARDSISOTROPY
Theideaofisotropycomesfromthepartitionfunctiondefinedin(Aroraetal.,2016),
Z(c)= (cid:88) exp (cid:0) c(cid:62)v(w) (cid:1) ,
w∈V
whereZ(c)shouldapproximatelybeaconstantwithanyunitvectorc(c.f. Lemma2.1in(Arora
etal.,2016)). Hence,wemathematicallydefineameasureofisotropyasfollows,
min Z(c)
(cid:107)c(cid:107)=1
I({v(w)})= , (1)
max Z(c)
(cid:107)c(cid:107)=1
where I({v(w)}) ranges from 0 to 1, and I({v(w)}) closer to 1 indicates that {v(w)} is more
isotropic. The intuition behind our postprocessing algorithm can also be motivated by letting
I({v(w)})→1.
LetV bethematrixstackedbyallwordvectors,wheretherowscorrespondtowordvectors,and
1 bethe|V|-dimensionalvectorswithallentriesequaltoone,Z(c)canbeequivalentlydefinedas
|V|
follows,
∞
1 (cid:88) 1 (cid:88)
Z(c)=|V|+1V|(cid:62)Vc+ c(cid:62)V(cid:62)Vc+ (c(cid:62)v(w))k.
| 2 k!
k=3 w∈V
I({v(w)})is,therefore,canbeverycoarselyapproximatedby,
• Afirstorderapproximation:
|V|+min 1(cid:62) Vc |V|−(cid:107)1(cid:62) V(cid:107)
(cid:107)c(cid:107)=1 |V| |V|
I({v(w)})≈ = .
|V|+max 1(cid:62) Vc |V|+(cid:107)1(cid:62) V(cid:107)
(cid:107)c(cid:107)=1 |V| |V|
LettingI({v(w)})=1yields(cid:107)1(cid:62) V(cid:107)=0,whichisequivalentto (cid:80) v(w)=0. The
|V| w∈V
intuitionbehindthefirstorderapproximationmatcheswiththefirststepoftheproposed
algorithm,whereweenforcev(w)tohaveazeromean.
• Asecondorderapproximation:
|V|+min 1(cid:62) Vc+min 1c(cid:62)V(cid:62)Vc |V|−(cid:107)1(cid:62) V(cid:107)+ 1σ2
I({v(w)})≈ (cid:107)c(cid:107)=1 |V| (cid:107)c(cid:107)=1 2 = |V| 2 min ,
|V|+max 1(cid:62) Vc+max 1c(cid:62)V(cid:62)Vc |V|+(cid:107)1(cid:62) V(cid:107)+ 1σ2
(cid:107)c(cid:107)=1 |V| (cid:107)c(cid:107)=1 2 |V| 2 max
whereσ andσ arethesmallestandlargestsingularvalueofV,respectively. Letting
min max
I({v(w)})=1yields(cid:107)1(cid:62) V(cid:107)=0andσ =σ . Thefactthatσ =σ suggests
|V| min max min max
thespectrumofv(w)’sshouldbeflat. Thesecondstepoftheproposedalgorithmremoves
thehighestsingularvalues,andthereforeexplicitlyflattenthespectrumofV.
Empirical Verification Indeed, we empirically validate the effect of postprocessing of on
I({v(w)}). Since there is no closed-form solution for argmax Z(c) or argmin Z(c),
(cid:107)c(cid:107)=1 (cid:107)c(cid:107)=1
anditisimpossibletoenumerateallc’s,weestimatethemeasureby,
min Z(c)
I({v(w)})≈ c∈C ,
max Z(c)
c∈C
whereC isthesetofeigenvectorsofV(cid:62)V. ThevalueofI({v(w)})fortheoriginalvectorsand
processed ones are reported in Table 2, where we can observe that the degree of isotropy vastly
increasesintermsofthismeasure.
5

PublishedasaconferencepaperatICLR2018
|     | before after | Aformalwaytoverifytheisotropypropertyisto |     |     |     |
| --- | ------------ | ----------------------------------------- | --- | --- | --- |
WORD2VEC 0.7 0.95 directlycheckifthe“self-normalization"property
(i.e.,Z(c)isaconstant,independentofc(Andreas
|     | GLOVE 0.065 | 0.6                             |     |     |            |
| --- | ----------- | ------------------------------- | --- | --- | ---------- |
|     |             | &Klein,2015))holdsmorestrongly. |     |     | Suchavali- |
Table2:Before-Afteronthemeasureofisotropy. dationisseendiagrammaticallyinFigure3where
werandomlysampled1,000c’sas(Aroraetal.,
2016).
|     | 1000                                 |           | 1000                    |             |         |
| --- | ------------------------------------ | --------- | ----------------------- | ----------- | ------- |
|     | before                               |           | before                  |             |         |
|     | 800 after                            |           | 800 after               |             |         |
|     | ycneuqerf                            | ycneuqerf |                         |             |         |
|     | 600                                  |           | 600                     |             |         |
|     | 400                                  |           | 400                     |             |         |
|     | 200                                  |           | 200                     |             |         |
|     | 0                                    |           | 0                       |             |         |
|     | 0.800.850.900.951.001.051.101.151.20 |           | 0.7 0.8                 | 0.9 1.0 1.1 | 1.2 1.3 |
|     | partition function Z(c)              |           | partition function Z(c) |             |         |
|     | (a) word2vec                         |           | (b)                     | GloVe       |         |
Figure3: ThehistogramofZ(c)for1,000randomlysampledvectorscofunitnorm,wherex-axisis
normalizedbythemeanofallvaluesandD =2forGLOVEandD =3forWORD2VEC.
3 EXPERIMENTS
Given the popularity and widespread use of WORD2VEC (Mikolov et al., 2013) and GLOVE
(Penningtonetal.,2014),weusetheirpubliclyavailablepre-trainedreprepsentationsinthefollowing
experiments. We choose D = 3 for WORD2VEC and D = 2 for GLOVE. The key underlying
principle behind word representations is that similar words should have similar representations.
Following the tradition of evaluating word representations (Schnabel et al., 2015; Baroni et al.,
2014),weperformthreecanonicallexical-leveltasks: (a)wordsimilarity;(b)conceptcategorization;
(c) word analogy; and one sentence-level task: (d) semantic textual similarity. The processed
representationsconsistentlyimproveperformanceonallthreeofthem,andespeciallystronglyonthe
firsttwo.
|     |                | WordSimilarity |     | Thewordsimilaritytaskisas |     |
| --- | -------------- | -------------- | --- | ------------------------- | --- |
|     | WORD2VEC GLOVE |                |     |                           |     |
follows: givenapairofwords,thealgorithmas-
|      | orig. proc. orig. | proc. |                |            |                   |
| ---- | ----------------- | ----- | -------------- | ---------- | ----------------- |
|      |                   | signs | a “similarity" | score – if | the pair of words |
| RG65 | 76.08 78.34 76.96 | 74.36 |                |            |                   |
WS 68.29 69.05 73.79 76.79 are highly related then the score should also be
|     |                   | highandviceversa. |     | Thealgorithmisevaluatedin |     |
| --- | ----------------- | ----------------- | --- | ------------------------- | --- |
| RW  | 53.74 54.33 46.41 | 52.04             |     |                           |     |
termsofSpearman’srankcorrelationcomparedto
| MEN | 78.20 79.08 80.49 | 81.78 |     |     |     |
| --- | ----------------- | ----- | --- | --- | --- |
(agoldsetof)humanjudgements.
| MTurk | 68.23 69.35 69.29 | 70.85 |     |     |     |
| ----- | ----------------- | ----- | --- | --- | --- |
SimLex 44.20 45.10 4083 44.97 For this experiment, we use seven standard
SimVerb 36.35 36.50 28.33 32.23 datasets: thefirstpublishedRG65dataset(Ruben-
|     |     | stein | & Goodenough, | 1965); | the widely used |
| --- | --- | ----- | ------------- | ------ | --------------- |
Table 3: Before-After results (x100) on word WordSim-353 (WS) dataset (Finkelstein et al.,
similaritytaskonsevendatasets. 2001)whichcontains353pairsofcommonlyused
verbsandnouns;therare-words(RW)dataset(Lu-
ongetal.,2013)composedofrarelyusedwords;theMENdataset(Brunietal.,2014)wherethe3000
pairsofwordsareratedbycrowdsourcedparticipants;theMTurkdataset(Radinskyetal.,2011)
wherethe287pairsofwordsareratedintermsofrelatedness;theSimLex-999(SimLex)dataset(Hill
etal.,2016)wherethescoremeasures“genuine"similarity;andlastlytheSimVerb-3500(SimVerb)
dataset(Gerzetal.,2016),anewlyreleasedlargedatasetfocusingonsimilarityofverbs.
Inourexperiment,thealgorithmscoresthesimilaritybetweentwowordsbythecosinesimilarity
between the two corresponding word vectors (CosSim(v ,v ) = v(cid:62)v /(cid:107)v (cid:107)(cid:107)v (cid:107)). The detailed
|     |     |     | 1 2 | 1 2 1 | 2   |
| --- | --- | --- | --- | ----- | --- |
performance on the seven datasets is reported in Table 3, where we see a consistent and signifi-
cant performance improvement due to postprocessing, across all seven datasets. These statistics
6

PublishedasaconferencepaperatICLR2018
(averageimprovementof2.3%)suggestthatbyremovingthecommonparts,theremainingword
representationsareabletocapturestrongersemanticrelatedness/similaritybetweenwords.
|        |          |       |       |       | Concept                             | Categorization |      | This     | task is an  | indirect  |
| ------ | -------- | ----- | ----- | ----- | ----------------------------------- | -------------- | ---- | -------- | ----------- | --------- |
|        | WORD2VEC |       | GLOVE |       |                                     |                |      |          |             |           |
|        |          |       |       |       | evaluationofthesimilarityprinciple: |                |      |          | givenasetof |           |
|        | orig.    | proc. | orig. | proc. |                                     |                |      |          |             |           |
|        |          |       |       |       | concepts,                           | the algorithm  |      | needs    | to group    | them into |
|        | ap 54.43 | 57.72 | 64.18 | 65.42 |                                     |                |      |          |             |           |
|        |          |       |       |       | different categories                |                | (for | example, | “bear”      | and “cat” |
| esslli | 75.00    | 84.09 | 81.82 | 81.82 |                                     |                |      |          |             |           |
arebothanimalsand“city”and“country”areboth
| battig | 71.97           | 81.71   | 86.59  | 86.59  |                |             |       |            |             |         |
| ------ | --------------- | ------- | ------ | ------ | -------------- | ----------- | ----- | ---------- | ----------- | ------- |
|        |                 |         |        |        | related to     | districts). | The   | clustering | performance | is      |
|        |                 |         |        |        | then evaluated | in          | terms | of purity  | (Manning    | et al., |
| Table  | 4: Before-After | results | (x100) | on the |                |             |       |            |             |         |
2008)–thefractionofthetotalnumberoftheobjects
categorizationtask.
thatwereclassifiedcorrectly.
Weconductthistaskonthreedifferentdatasets: theAlmuhareb-Poesio(ap)dataset(Almuhareb,
2006)contains402conceptswhichfallinto21categories;theESSLLI2008DistributionalSemantic
Workshopshared-taskdataset(Baronietal.,2008)thatcontains44conceptsin6categories;andthe
Battigtestset(Baroni&Lenci,2010)thatcontains83wordsin10categories.
Herewefollowthesettingandtheproposedalgorithmin(Baronietal.,2014;Schnabeletal.,2015)
– we cluster words (via their representations) using the classical k-Means algorithm (with fixed
k). Again, the processed vectors perform consistently better on all three datasets (with average
improvementof2.5%);thefulldetailsareinTable4.
|     |     |          |       |     | WordAnalogy |     | Theanalogytaskteststowhat |     |     |     |
| --- | --- | -------- | ----- | --- | ----------- | --- | ------------------------- | --- | --- | --- |
|     |     | WORD2VEC | GLOVE |     |             |     |                           |     |     |     |
extentthewordrepresentationscanencodelatent
|     |     | orig. proc. | orig. | proc. |     |     |     |     |     |     |
| --- | --- | ----------- | ----- | ----- | --- | --- | --- | --- | --- | --- |
syntax 73.46 73.50 74.95 75.40 linguisticrelationsbetweenapairofwords.Given
|           |     |             |       |       | threewordsw                      |     | ,w ,andw |     | ,theanalogytaskre- |          |
| --------- | --- | ----------- | ----- | ----- | -------------------------------- | --- | -------- | --- | ------------------ | -------- |
| semantics |     | 72.28 73.36 | 79.22 | 79.25 |                                  |     | 1 2      | 3   |                    |          |
|           |     |             |       |       | quiresthealgorithmtofindthewordw |     |          |     |                    | suchthat |
|           | all | 72.93 73.44 | 76.89 | 77.15 |                                  |     |          |     |                    | 4        |
|           |     |             |       |       | w istow                          | asw | istow    | .   |                    |          |
|           |     |             |       |       | 4                                | 3   | 2        | 1   |                    |          |
Table5: Before-Afterresults(x100)ontheword
Weusetheanalogydatasetintroducedin(Mikolov
analogytask.
|     |     |     |     |     | etal.,2013).                                |     | Thedatasetcanbedividedintotwo |     |     |     |
| --- | --- | --- | --- | --- | ------------------------------------------- | --- | ----------------------------- | --- | --- | --- |
|     |     |     |     |     | parts: (a)thesemanticpartcontainingaround9k |     |                               |     |     |     |
questions, focusing on the latent semantic relation between pairs of words (for example, what is
toChicagoasTexasistoHouston); and(b)thesyntaticonecontainingroughly10.5kquestions,
focusingonthelatentsyntaticrelationbetweenpairsofwords(forexample,whatisto“amazing”as
“apprently”isto“apparent”).
Inoursetting,weusetheoriginalalgorithmintroducedin(Mikolovetal.,2013)tosolvethisproblem,
i.e.,w isthewordthatmaximizethecosinesimilaritybetweenv(w )andv(w )−v(w )+v(w ).
|     | 4   |     |     |     |     |     | 4   |     | 2 1 | 3   |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
TheaverageperformanceontheanalogytaskisprovidedinTable5(withadetailedperformance
providedinTable19inAppendixD).Itcanbenoticedthatwhilepostprocessingcontinuestoimprove
theperformance,theimprovementisnotaspronouncedasearlier. Wehypothesizethatthisisbecause
themeanandsomedominantcomponentsgetcanceledduringthesubtractionofv(w )fromv(w ),
|     |     |     |     |     |     |     |     |     | 2   | 1   |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
andthereforetheeffectofpostprocessingislessrelevant.
|     |          |     |       |     | Semantic | Textual | Similarity |     | Extrinsic | evalua- |
| --- | -------- | --- | ----- | --- | -------- | ------- | ---------- | --- | --------- | ------- |
|     | WORD2VEC |     | GLOVE |     |          |         |            |     |           |         |
tionsmeasurethecontributionofawordrepresenta-
|      | orig. | proc. | orig. | proc. |                                |     |     |     |               |     |
| ---- | ----- | ----- | ----- | ----- | ------------------------------ | --- | --- | --- | ------------- | --- |
|      |       |       |       |       | tiontospecificdownstreamtasks; |     |     |     | below,westudy |     |
| 2012 | 57.22 | 57.67 | 48.27 | 54.06 |                                |     |     |     |               |     |
theeffectofpostprocessingonastandardsentence
| 2013 | 56.81 | 57.98 | 44.83 | 57.71 |          |        |          |         |            |       |
| ---- | ----- | ----- | ----- | ----- | -------- | ------ | -------- | ------- | ---------- | ----- |
|      |       |       |       |       | modeling | task – | semantic | textual | similarity | (STS) |
| 2014 | 62.89 | 63.30 | 51.11 | 59.23 |          |        |          |         |            |       |
whichaimsattestingthedegreetowhichthealgo-
| 2015 | 62.74 | 63.35 | 47.23 | 57.29 |     |     |     |     |     |     |
| ---- | ----- | ----- | ----- | ----- | --- | --- | --- | --- | --- | --- |
rithmcancapturethesemanticequivalencebetween
| SICK | 70.10     | 7020  | 65.14 | 67.85 |               |                                 |     |     |     |     |
| ---- | --------- | ----- | ----- | ----- | ------------- | ------------------------------- | --- | --- | --- | --- |
|      |           |       |       |       | twosentences. | Foreachpairofsentences,thealgo- |     |     |     |     |
|      | all 60.88 | 61.45 | 49.19 | 56.76 |               |                                 |     |     |     |     |
rithmneedstomeasurehowsimilarthetwosentences
|       |                 |         |        |        | are. Thedegreetowhichthemeasurematcheswith |     |     |     |     |     |
| ----- | --------------- | ------- | ------ | ------ | ------------------------------------------ | --- | --- | --- | --- | --- |
| Table | 6: Before-After | results | (x100) | on the |                                            |     |     |     |     |     |
humanjudgment(intermsofPearsoncorrelation)is
semantictextualsimilaritytasks.
|     |     |     |     |     | anindexofthealgorithm’sperformance. |     |     |     | Wetestthe |     |
| --- | --- | --- | --- | --- | ----------------------------------- | --- | --- | --- | --------- | --- |
wordrepresentationson20textualsimilaritydatasets
7

PublishedasaconferencepaperatICLR2018
from the 2012-2015 SemEval STS tasks (Agirre et al., 2012; 2013; 2014; 2015), and the 2012
SemEvalSemanticRelatedtask(SICK)(Marellietal.,2014).
Representing sentences by the average of their constituent word representations is surprisingly
effectiveinencodingthesemanticinformationofsentences(Wietingetal.,2015;Adietal.,2016)
andclosetothestate-of-the-artinthesedatasets. Wefollowthisrubricandrepresentasentence
s based on its averaged word representation, i.e., v(s) = 1 (cid:80) v(w), and then compute the
|s| w∈s
similarity between two sentences via the cosine similarity between the two representations. The
averageperformanceoftheoriginalandprocessedrepresentationsisitemizedinTable6(witha
detailedperformanceinTable20inAppendixE)–weseeaconsistentandsignificantimprovement
inperformancebecauseofpostprocessing(onaverage4%improvement).
4 POSTPROCESSING AND SUPERVISED CLASSIFICATION
SuperviseddownstreamNLPapplicationshavegreatlyimprovedtheirperformancesinrecentyears
bycombiningthediscriminativelearningpowersofneuralnetworksinconjunctionwiththeword
representations.Weevaluatetheperformanceofavarietyofneuralnetworkarchitecturesonastandard
and important NLP application: text classification, with sentiment analysis being a particularly
importantandpopularexample. Thetaskisdefinedasfollows: givenasentence,thealgorithmneeds
todecidewhichcategoryitfallsinto. Thecategoriescanbeeitherbinary(e.g.,positive/negative)or
canbemorefine-grained(e.g. verypositive,positive,neutral,negative,andverynegative).
Weevaluatethewordrepresentations(withandwithoutpostprocessing)usingfourdifferentneural
networkarchitectures(CNN,vanilla-RNN,GRU-RNNandLSTM-RNN)onfivebenchmarks: (a)the
moviereview(MR)dataset(Pang&Lee,2005);(b)thesubjectivity(SUBJ)dataset(Pang&Lee,
2004);(c)theTRECquestiondataset(Li&Roth,2002);(d)theIMDbdataset(Maasetal.,2011);
(e)thestanfordsentimenttreebank(SST)dataset(Socheretal.,2013a). Adetaileddescriptionof
thesestandarddatasets,theirtraining/testparametersandthecrossvalidationmethodsadoptedis
inAppendixF. Specifically,weallowtheparameterD(i.e.,thenumberofnulledcomponents)to
varybetween0and4,andthebestperformanceofthefourneuralnetworkarchitectureswiththe
now-standardCNN-basedtextclassificationalgorithm(Kim,2014)(implementedusingtensorflow4)
isitemizedinTable7. Thekeyobservationisthattheperformanceofpostprocessingisbetterina
majority(34outof40)oftheinstancesby2.32%onaverage,andintheresttheinstancesthetwo
performances(withandwithoutpostprocessing)arecomparable.
CNN vanilla-RNN GRU-RNN LSTM-RNN
WORD2VEC GLOVE WORD2VEC GLOVE WORD2VEC GLOVE WORD2VEC GLOVE
orig. proc. orig. proc. orig. proc. orig. proc. orig. proc. orig. proc. orig. proc. orig. proc.
MR 70.80 71.27 71.01 71.11 74.95 74.01 71.14 72.56 77.86 78.26 74.98 75.13 75.69 77.34 72.02 71.84
SUBJ 87.14 87.33 86.98 87.25 82.85 87.60 81.45 87.37 90.96 91.10 91.16 91.85 90.23 90.54 90.74 90.82
TREC 87.80 89.00 87.60 89.00 80.60 89.20 85.20 89.00 91.60 92.40 91.60 93.00 88.00 91.20 85.80 91.20
SST 38.46 38.33 38.82 37.83 42.08 39.91 41.45 41.90 41.86 45.02 36.52 37.69 43.08 42.08 37.51 38.05
IMDb 86.68 87.12 87.27 87.10 50.15 53.14 52.76 76.07 82.96 83.47 81.50 82.44 81.29 82.60 79.10 81.33
Table7:Before-Afterresults(x100)onthetextclassificationtaskusingCNN(Kim,2014)andvanilla
RNN,GRU-RNNandLSTM-RNN.
A further validation of the postprocessing operation in a variety of downstream applications (eg:
named entity recognition, syntactic parsers, machine translation) and classification methods (eg:
randomforests,neuralnetworkarchitectures)isofactiveresearchinterest. Ofparticularinterestis
theimpactofthepostprocessingontherateofconvergenceandgeneralizationcapabilitiesofthe
classifiers. Suchasystematicstudywouldentailaconcertedandlarge-scaleeffortbytheresearch
communityandislefttofutureresearch.
Discussion Allneuralnetworkarchitectures,rangingfromfeedforwardtorecurrent(eithervanilla
orGRUorLSTM),implementatleastlinearprocessingofhidden/inputstatevectorsateachoftheir
nodes;thusthepostprocessingoperationsuggestedinthispapercaninprinciplebeautomatically
“learnt”bytheneuralnetwork,ifsuchinternallearningisin-linewiththeend-to-endtrainingexamples.
Yet,inpracticethisiscomplicatedduetolimitationsofoptimizationprocedures(SGD)andsample
4https://github.com/dennybritz/cnn-text-classification-tf
8

PublishedasaconferencepaperatICLR2018
noise. WeconductapreliminaryexperimentinAppendixBandshowthatsubtractingthemean(i.e.,
thefirststepofpostprocessing)is“effectivelylearnt"byneuralnetworkswithintheirnodes.
5 CONCLUSION
Wepresentasimplepostprocessingoperationthatrenderswordrepresentationsevenstronger,by
eliminatingthetopprincipalcomponentsofallwords. Suchansimpleoperationcouldbeusedfor
wordembeddingsindownstreamtasksorasintializationsfortrainingtask-specificembeddings. Due
totheirpopularity,wehaveusedthepublishedrepresentationsofWORD2VECandGLOVEinEn-
glishinthemaintextofthispaper;postprocessingcontinuestobesuccessfulforotherrepresentations
andinmultilingualsettings–thedetailedempiricalresultsaretabulatedinAppendixC.
REFERENCES
YossiAdi,EinatKermany,YonatanBelinkov,OferLavi,andYoavGoldberg. Fine-grainedanalysis
ofsentenceembeddingsusingauxiliarypredictiontasks. arXivpreprintarXiv:1608.04207,2016.
EnekoAgirre,MonaDiab,DanielCer,andAitorGonzalez-Agirre. Semeval-2012task6: Apilot
on semantic textual similarity. In Proceedings of the First Joint Conference on Lexical and
ComputationalSemantics-Volume1: Proceedingsofthemainconferenceandthesharedtask,and
Volume2: ProceedingsoftheSixthInternationalWorkshoponSemanticEvaluation,pp.385–393.
AssociationforComputationalLinguistics,2012.
EnekoAgirre,DanielCer,MonaDiab,AitorGonzalez-Agirre,andWeiweiGuo. sem2013shared
task: Semantictextualsimilarity,includingapilotontyped-similarity. InIn*SEM2013: The
SecondJointConferenceonLexicalandComputationalSemantics.AssociationforComputational
Linguistics.Citeseer,2013.
EnekoAgirre,CarmenBanea,ClaireCardie,DanielCer,MonaDiab,AitorGonzalez-Agirre,Weiwei
Guo,RadaMihalcea,GermanRigau,andJanyceWiebe. Semeval-2014task10: Multilingualse-
mantictextualsimilarity. InProceedingsofthe8thinternationalworkshoponsemanticevaluation
(SemEval2014),pp.81–91,2014.
EnekoAgirre,CarmenBaneab,ClaireCardiec,DanielCerd,MonaDiabe,AitorGonzalez-Agirrea,
WeiweiGuof,InigoLopez-Gazpioa,MontseMaritxalara,RadaMihalceab,etal. Semeval-2015
task2: Semantictextualsimilarity,english,spanishandpilotoninterpretability. InProceedingsof
the9thinternationalworkshoponsemanticevaluation(SemEval2015),pp.252–263,2015.
RamiAl-Rfou,BryanPerozzi,andStevenSkiena. Polyglot: Distributedwordrepresentationsfor
multilingualnlp. arXivpreprintarXiv:1307.1662,2013.
AbdulrahmanAlmuhareb. Attributesinlexicalacquisition. PhDthesis,UniversityofEssex,2006.
JacobAndreasandDan Klein. Whenandwhyare log-linearmodelsself-normalizing? InHLT-
NAACL,pp.244–249,2015.
SanjeevArora,YuanzhiLi,YingyuLiang,TengyuMa,andAndrejRisteski. Alatentvariablemodel
approach to pmi-based word embeddings. Transactions of the Association for Computational
Linguistics, 4:385–399, 2016. ISSN 2307-387X. URL https://transacl.org/ojs/
index.php/tacl/article/view/742.
SanjeevArora,YingyuLiang,andTengyuMa. Asimplebuttough-to-beatbaselineforsentence
embeddings. InInternationalConferenceonLearningRepresentations.,2017.
DzmitryBahdanau,KyunghyunCho,andYoshuaBengio. Neuralmachinetranslationbyjointly
learningtoalignandtranslate. arXivpreprintarXiv:1409.0473,2014.
M Baroni, S Evert, and A Lenci. Bridging the gap between semantic theory and computational
simulations: Proceedingsoftheesslliworkshopondistributionallexicalsemantics. Hamburg,
Germany: FOLLI,2008.
9

PublishedasaconferencepaperatICLR2018
MarcoBaroniandAlessandroLenci. Distributionalmemory: Ageneralframeworkforcorpus-based
semantics. ComputationalLinguistics,36(4):673–721,2010.
Marco Baroni, Georgiana Dinu, and Germán Kruszewski. Don’t count, predict! a systematic
comparisonofcontext-countingvs.context-predictingsemanticvectors. InACL(1),pp.238–247,
2014.
Yoshua Bengio, Réjean Ducharme, Pascal Vincent, and Christian Jauvin. A neural probabilistic
languagemodel. journalofmachinelearningresearch,3(Feb):1137–1155,2003.
AntoineBordes, NicolasUsunier, AlbertoGarcia-Duran, JasonWeston, andOksanaYakhnenko.
Translatingembeddingsformodelingmulti-relationaldata. InAdvancesinNeuralInformation
ProcessingSystems,pp.2787–2795,2013.
Michael W Browne. The maximum-likelihood solution in inter-battery factor analysis. British
JournalofMathematicalandStatisticalPsychology,32(1):75–86,1979.
EliaBruni,Nam-KhanhTran,andMarcoBaroni. Multimodaldistributionalsemantics. J.Artif.Intell.
Res.(JAIR),49(1-47),2014.
JohnABullinariaandJosephPLevy. Extractingsemanticrepresentationsfromwordco-occurrence
statistics: stop-lists,stemming,andsvd. Behaviorresearchmethods,44(3):890–907,2012.
JoséCamacho-Collados, MohammadTaherPilehvar, andRobertoNavigli. Aframeworkforthe
constructionofmonolingualandcross-lingualwordsimilaritydatasets. InACL(2),pp.1–7,2015.
JunyoungChung,CaglarGülçehre,KyunghyunCho,andYoshuaBengio. Gatedfeedbackrecurrent
neuralnetworks. InICML,pp.2067–2075,2015.
RonanCollobert,JasonWeston,LéonBottou,MichaelKarlen,KorayKavukcuoglu,andPavelKuksa.
Naturallanguageprocessing(almost)fromscratch. JournalofMachineLearningResearch,12
(Aug):2493–2537,2011.
ParamveerDhillon,JordanRodu,DeanFoster,andLyleUngar. Twostepcca: Anewspectralmethod
forestimatingvectormodelsofwords. arXivpreprintarXiv:1206.6403,2012.
LevFinkelstein,EvgeniyGabrilovich,YossiMatias,EhudRivlin,ZachSolan,GadiWolfman,and
Eytan Ruppin. Placing search in context: The concept revisited. In Proceedings of the 10th
internationalconferenceonWorldWideWeb,pp.406–414.ACM,2001.
DanielaGerz,IvanVulic´,FelixHill,RoiReichart,andAnnaKorhonen. Simverb-3500: Alarge-scale
evaluationsetofverbsimilarity. arXivpreprintarXiv:1608.00869,2016.
KlausGreff,RupeshKSrivastava,JanKoutník,BasRSteunebrink,andJürgenSchmidhuber. Lstm:
Asearchspaceodyssey. IEEEtransactionsonneuralnetworksandlearningsystems,2016.
Felix Hill, Roi Reichart, and Anna Korhonen. Simlex-999: Evaluating semantic models with
(genuine)similarityestimation. ComputationalLinguistics,2016.
HaroldHotelling. Relationsbetweentwosetsofvariates. Biometrika,28(3/4):321–377,1936.
Eric H Huang, Richard Socher, Christopher D Manning, and Andrew Y Ng. Improving word
representationsviaglobalcontextandmultiplewordprototypes. InProceedingsofthe50thAnnual
MeetingoftheAssociationforComputationalLinguistics: LongPapers-Volume1,pp.873–882.
AssociationforComputationalLinguistics,2012.
YoonKim.Convolutionalneuralnetworksforsentenceclassification.arXivpreprintarXiv:1408.5882,
2014.
BeatriceLaurentandPascalMassart.Adaptiveestimationofaquadraticfunctionalbymodelselection.
AnnalsofStatistics,pp.1302–1338,2000.
OmerLevyandYoavGoldberg.Neuralwordembeddingasimplicitmatrixfactorization.InAdvances
inneuralinformationprocessingsystems,pp.2177–2185,2014.
10

PublishedasaconferencepaperatICLR2018
XinLiandDanRoth. Learningquestionclassifiers. InProceedingsofthe19thinternationalconfer-
enceonComputationallinguistics-Volume1,pp.1–7.AssociationforComputationalLinguistics,
2002.
Thang Luong, Richard Socher, and Christopher D Manning. Better word representations with
recursiveneuralnetworksformorphology. InCoNLL,pp.104–113,2013.
Andrew L Maas, Raymond E Daly, Peter T Pham, Dan Huang, Andrew Y Ng, and Christopher
Potts. Learningwordvectorsforsentimentanalysis. InProceedingsofthe49thAnnualMeeting
oftheAssociationforComputationalLinguistics: HumanLanguageTechnologies-Volume1,pp.
142–150.AssociationforComputationalLinguistics,2011.
ChristopherDManning,PrabhakarRaghavan,HinrichSchütze,etal. Introductiontoinformation
retrieval,volume1. CambridgeuniversitypressCambridge,2008.
MarcoMarelli,StefanoMenini,MarcoBaroni,LuisaBentivogli,RaffaellaBernardi,andRoberto
Zamparelli. Asickcurefortheevaluationofcompositionaldistributionalsemanticmodels. In
LREC,pp.216–223,2014.
TomasMikolov,MartinKarafiát,LukasBurget,JanCernocky`,andSanjeevKhudanpur. Recurrent
neuralnetworkbasedlanguagemodel. InInterspeech,volume2,pp. 3,2010.
TomasMikolov,KaiChen,GregCorrado,andJeffreyDean. Efficientestimationofwordrepresenta-
tionsinvectorspace. arXivpreprintarXiv:1301.3781,2013.
AndriyMnihandGeoffreyHinton. Threenewgraphicalmodelsforstatisticallanguagemodelling. In
Proceedingsofthe24thinternationalconferenceonMachinelearning,pp.641–648.ACM,2007.
BoPangandLillianLee. Asentimentaleducation: Sentimentanalysisusingsubjectivitysumma-
rizationbasedonminimumcuts. InProceedingsofthe42ndannualmeetingonAssociationfor
ComputationalLinguistics,pp.271.AssociationforComputationalLinguistics,2004.
BoPangandLillianLee. Seeingstars: Exploitingclassrelationshipsforsentimentcategorization
with respect to rating scales. In Proceedings of the 43rd annual meeting on association for
computationallinguistics,pp.115–124.AssociationforComputationalLinguistics,2005.
JeffreyPennington,RichardSocher,andChristopherDManning. Glove: Globalvectorsforword
representation. InEMNLP,volume14,pp.1532–43,2014.
AlkesLPrice,NickJPatterson,RobertMPlenge,MichaelEWeinblatt,NancyAShadick,andDavid
Reich. Principalcomponentsanalysiscorrectsforstratificationingenome-wideassociationstudies.
Naturegenetics,38(8):904–909,2006.
Kira Radinsky, Eugene Agichtein, Evgeniy Gabrilovich, and Shaul Markovitch. A word at a
time: computingwordrelatednessusingtemporalsemanticanalysis. InProceedingsofthe20th
internationalconferenceonWorldwideweb,pp.337–346.ACM,2011.
HerbertRubensteinandJohnBGoodenough. Contextualcorrelatesofsynonymy. Communications
oftheACM,8(10):627–633,1965.
MagnusSahlgren,AmaruCubaGyllensten,FredrikEspinoza,OlaHamfors,JussiKarlgren,Fredrik
Olsson, Per Persson, Akshay Viswanathan, and Anders Holst. The gavagai living lexicon. In
NicolettaCalzolari(ConferenceChair),KhalidChoukri,ThierryDeclerck,SaraGoggi,Marko
Grobelnik,BenteMaegaard,JosephMariani,HeleneMazo,AsuncionMoreno,JanOdijk,and
SteliosPiperidis(eds.),ProceedingsoftheTenthInternationalConferenceonLanguageResources
andEvaluation(LREC2016),Paris,France,may2016.EuropeanLanguageResourcesAssociation
(ELRA). ISBN978-2-9517408-9-1.
Tobias Schnabel, Igor Labutov, David Mimno, and Thorsten Joachims. Evaluation methods for
unsupervisedwordembeddings. InProc.ofEMNLP,2015.
RichardSocher,DanqiChen,ChristopherDManning,andAndrewNg. Reasoningwithneuraltensor
networksforknowledgebasecompletion. InAdvancesinNeuralInformationProcessingSystems,
pp.926–934,2013a.
11

PublishedasaconferencepaperatICLR2018
RichardSocher,AlexPerelygin,JeanYWu,JasonChuang,ChristopherDManning,AndrewYNg,
andChristopherPotts. Recursivedeepmodelsforsemanticcompositionalityoverasentiment
treebank. InProceedingsoftheconferenceonempiricalmethodsinnaturallanguageprocessing
(EMNLP),volume1631,pp.1642.Citeseer,2013b.
KarlStratos,MichaelCollins,andDanielHsu. Model-basedwordembeddingsfromdecompositions
ofcountmatrices. InProceedingsofACL,pp.1282–1291,2015.
IlyaSutskever,OriolVinyals,andQuocVLe. Sequencetosequencelearningwithneuralnetworks.
InAdvancesinneuralinformationprocessingsystems,pp.3104–3112,2014.
JosephTurian,LevRatinov,andYoshuaBengio. Wordrepresentations: asimpleandgeneralmethod
forsemi-supervisedlearning. InProceedingsofthe48thannualmeetingoftheassociationfor
computationallinguistics,pp.384–394.AssociationforComputationalLinguistics,2010.
JohnWieting,MohitBansal,KevinGimpel,KarenLivescu,andDanRoth. Fromparaphrasedatabase
tocompositionalparaphrasemodelandback. arXivpreprintarXiv:1506.03487,2015.
Torsten Zesch and Iryna Gurevych. Automatically creating datasets for measures of semantic
relatedness. InProceedingsoftheWorkshoponLinguisticDistances,pp.16–24.Associationfor
ComputationalLinguistics,2006.
12

PublishedasaconferencepaperatICLR2018
Appendix: All-but-the-Top: Simple and Effective postprocessing for
Word Representations
A ANGULAR ASYMMETRY OF REPRESENTATIONS
AmodernunderstandingofwordrepresentationsinvolveseitherPMI-based(includingword2vec
(Mikolovetal.,2010;Levy&Goldberg,2014)andGloVe(Penningtonetal.,2014))orCCA-based
spectralfactorizationapproaches. WhileCCA-basedspectralfactorizationmethodshavelongbeen
understoodfromaprobabilistic(i.e.,generativemodel)viewpoint(Browne,1979;Hotelling,1936)
and recently in the NLP context (Stratos et al., 2015), a corresponding effort for the PMI-based
methodshasonlyrecentlybeenconductedinaninspiredwork(Aroraetal.,2016).
(Aroraetal.,2016)proposeagenerativemodel(namedRAND-WALK)ofsentences,whereevery
wordisparameterizedbyad-dimensionalvector. Withakeypostulatethatthewordvectorsare
angularly uniform (“isotropic"), the family of PMI-based word representations can be explained
undertheRAND-WALKmodelintermsofthemaximumlikelihoodrule. Ourobservationthatword
vectorslearntthroughPMI-basedapproachesarenotofzero-meanandarenotisotropic(c.f.Section
2)contradictswiththispostulate. TheisotropyconditionsarerelaxedinSection2.2of(Aroraetal.,
2016),butthematchwiththespectralpropertiesobservedinFigure1isnotimmediate.
Inthissection,weresolvethisbyexplicitlyrelaxingtheconstraintsonthewordvectorstodirectlyfit
theobservedspectralproperties. Therelaxedconditionsare: thewordvectorsshouldbeisotropic
aroundapoint(whosedistancetotheoriginisasmallfractionoftheaveragenormofwordvectors)
lying on a low dimensional subspace. Our main result is to show that even with this enlarged
parameter-space, the maximum likelihood rule continues to be close to the PMI-based spectral
factorizationmethods. Formally,themodel,theoriginalconstraintsof(Aroraetal.,2016)andthe
enlargedconstraintsonthewordvectorsarelistedbelow:
• Agenerativemodelofsentences: thewordattimet, denotedbyw , isgeneratedviaa
t
log-linearmodelwithalatentdiscoursevariablec (Aroraetal.,2016),i.e.,
t
p(w |c )= 1 exp (cid:0) c(cid:62)v(w ) (cid:1) , (2)
t t Z(c ) t t
t
wherev(w)∈Rd isthevectorrepresentationforawordwinthevocabularyV,c isthe
t
latentvariablewhichformsa“slowlymoving"randomwalk, andthepartitionfunction:
Z(c)= (cid:80) exp (cid:0) c(cid:62)v(w) (cid:1) .
w∈V
• Constraints on the word vectors: (Arora et al., 2016) suppose that there is a Bayesian
priorionthewordvectors:
The ensemble of word vectors consists of i.i.d. draws generated by v = s·vˆ,
where vˆ is from the spherical Gaussian distribution, and s is a scalar random
variable.
AdeterministicversionofthispriorisdiscussedinSection2.2of(Aroraetal.,2016),but
partofthese(relaxed)conditionsonthewordvectorsarespecificallymeantforTheorem4.1
andnotthemaintheorem(Theorem2.2). Thegeometryofthewordrepresentationsisonly
evaluatedviatheratioofthequadraticmeanofthesingularvaluestothesmallestonebeing
smallenough. Thismeetstherelaxedconditions, butnotsufficienttovalidatetheproof
approachofthemainresult(Theorem2.2);whatwouldbeneededisthattheratioofthe
largestsingularvaluetothesmallestonebesmall.
• Revisedconditions: WerevisetheBayesianpriorpostulate(andinadeterministicfashion)
formallyasfollows: thereisameanvectorµ,Dorthonormalvectorsu ,...,u (thatare
1 D
orthogonalandofunitnorm),suchthateverywordvectorv(w)canberepresentedby,
D
(cid:88)
v(w)=µ+ α (w)u +v˜(w), (3)
i i
i=1
where µ is bounded, α is bounded by A, D is bounded by DA2 = o(d), v˜(w) are
i
statisticallyisotropic. Bystatisticalisotropy,wemean: forhigh-dimensionalrectanglesR,
13

PublishedasaconferencepaperatICLR2018
|     | 1 (cid:80) |             |     | (cid:82) |                         |     |     |                           |     |
| --- | ---------- | ----------- | --- | -------- | ----------------------- | --- | --- | ------------------------- | --- |
|     |            | 1(v˜(w)∈R)→ |     |          | f(v˜)dv˜,as|V|→∞,wheref |     |     | isanangle-independentpdf, |     |
| |V| | w∈V        |             |     |          | R                       |     |     |                           |     |
i.e.,f(v˜)isafunctionof(cid:107)v˜(cid:107).
Therevisedpostulatediffersfromtheoriginaloneintwoways: (a)itimposesaformaldeterministic
constraint on the word vectors; (b) the revised postulate allows the word vectors to be angularly
asymmetric: aslongastheenergyinthedirectionofu ,...,u isbounded,thereisnoconstrainton
1 D
thecoefficients. Indeed,notethatthereisnoconstraintonv˜(w)tobeorthogonaltou ,...u .
1 D
Empirical Validation We can verify that the enlarged conditions are met by the existing word
representations. Specifically,thenaturalchoiceforµisthemeanofthewordrepresentationsand
u ...u arethesingularvectorsassociatedwiththetopDsingularvaluesofthematrixofword
1 D
vectors. WepickD =20forWORD2VECandD =10forGLOVE,andthecorrespondingvalueof
DA2forWORD2VECandGLOVEvectorsarebothroughly40,respectively;bothvaluesaresmall
comparedtod=300.
0.008
GLOVE
|     |     |     |     | 0.007 |     | WORD2VEC |     |     |     |
| --- | --- | --- | --- | ----- | --- | -------- | --- | --- | --- |
|     |     |     |     | 0.006 |     | random   |     |     |     |
oitar ecnairav
0.005
0.004
0.003
0.002
0.001
0.000
|     |     |     |     | 101 |     | 102 |     |     |     |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
index
Figure 4: Spectrum of the published WORD2VEC and GLOVE and random Gaussian matrices,
| ignoringthetopDcomponents;D |     |     |     | =10forGLOVEandD |     |     | =20forWORD2VEC. |     |     |
| --------------------------- | --- | --- | --- | --------------- | --- | --- | --------------- | --- | --- |
Thisleavesustocheckthestatisticalisotropyofthe“remaining"vectorsv˜(w)forwordswinthe
vocabulary. Wedothisbyplottingtheremainingspectrum(i.e. the(D+1)-th,...,300thsingular
values) for the published WORD2VEC and GLOVE vectors in Figure 4. As a comparison, the
empirical spectrum of a random Gaussian matrix is also plotted in Figure 4. We see that both
spectraareflat(sincethevocabularysizeismuchlargerthanthedimensiond = 300). Thusthe
postprocessingoperationcanalsobeviewedasawayofmakingthevectors“moreisotropic”.
MathematicalContribution Undertherevisedpostulate,weshowthatthemaintheoremin(Arora
| etal.,2016)(c.f.Theorem2.2)stillholds. |                                             |     |     |     | Formally: |     |      |     |     |
| -------------------------------------- | ------------------------------------------- | --- | --- | --- | --------- | --- | ---- | --- | --- |
| TheoremA.1                             | Supposethewordvectorssatisfytheconstraints. |     |     |     |           |     | Then |     |     |
)(cid:62)v(w
|     |       |     | d ef | p(w | ,w )   | v(w | )   |            |     |
| --- | ----- | --- | ---- | --- | ------ | --- | --- | ---------- | --- |
|     | PMI(w | ,w  | ) =  | log | 1 2 →  | 1   | 2   | , as|V|→∞, | (4) |
|     |       | 1   | 2    | p(w | )p(w ) |     | d   |            |     |
|     |       |     |      |     | 1 2    |     |     |            |     |
wherep(w)istheunigramdistributioninducedfromthemodel(2),andp(w ,w )istheprobability
1 2
| thattwowordsw |     | andw | occurwitheachotherwithindistanceq. |     |     |     |     |     |     |
| ------------- | --- | ---- | ---------------------------------- | --- | --- | --- | --- | --- | --- |
|               | 1   |      | 2                                  |     |     |     |     |     |     |
TheproofisinAppendixG.TheoremA.1suggeststhattheRAND-WALKgenerativemodelandits
propertiesproposedby(Aroraetal.,2016)canbegeneralizedtoabroadersetting(witharelaxed
restrictiononthegeometryofwordrepresentations)–relevantly,thisrelaxationonthegeometryof
wordrepresentationsisempiricallysatisfiedbythevectorslearntaspartofthemaximumlikelihood
rule.
| B NEURAL | NETWORKS |     |     | LEARN | TO POSTPROCESS |     |     |     |     |
| -------- | -------- | --- | --- | ----- | -------------- | --- | --- | --- | --- |
Everyneuralnetworkfamilypossesestheabilitytoconductlinearprocessinginsidetheirnodes;this
includesfeedforwardandrecurrentandconvolutionalneuralnetworkmodels. Thus,inprinciple,
thepostprocessingoperationcanbe“learntandimplemented"withintheparametersoftheneural
network. Ontheotherhand,duetothelargenumberofparameterswithintheneuralnetwork,it
14

PublishedasaconferencepaperatICLR2018
is unclear how to verify such a process, even if it were learnt (only one of the layers might be
implementingthepostprocessingoperationorviaacombinationofmultipleeffects).
To address this issue, we have adopted a comparative approach in the rest of this section. The
comparativeapproachinvolvesaddinganextralayerinterposedinbetweentheinputs(whichare
wordvectors)andtherestoftheneuralnetwork. Thisextralayerinvolvesonlylinearprocessing.
Next we compare the results of the final parameters of the extra layer (trained jointly with the
rest of tne neural network parameters, using the end-to-end training examples) with and without
preprocessingofthewordvectors. Suchacomparativeapproachallowsustoseparatetheeffectof
thepostprocessingoperationonthewordvectorsfromthecomplicated“semantics”oftheneural
networkparameters.
softmax
|     |            |              |            |      | vanilla     | GRU         | LSTM        |
| --- | ---------- | ------------ | ---------- | ---- | ----------- | ----------- | ----------- |
|     | h1         | h2           | hT         |      | W. G.       | W. G.       | W. G.       |
|     |            |              |            | MR   | 82.07 49.23 | 81.35 47.63 | 77.95 44.92 |
|     | nonlinear	 | nonlinear	 … | nonlinear	 |      |             |             |             |
|     | unit       | unit         | unit       | SUBJ | 84.02 49.94 | 83.15 50.60 | 83.39 48.95 |
v(w1)   b v(w2)   b v(wT)   b TREC 81.68 52.99 82.68 50.42 80.80 46.77
|     |       |             |       | SST  | 79.64 46.59 | 78.06 43.21 | 77.72 42.82 |
| --- | ----- | ----------- | ----- | ---- | ----------- | ----------- | ----------- |
|     |       | linear	bias |       | IMDb | 93.48 66.37 | 94.49 55.24 | 87.27 46.74 |
|     | v(w1) | v(w2) …     | v(wT) |      |             |             |             |
Figure6: Thecosinesimilarity(x100)between
|                                          |     |     |     | b     | +µandb | ,whereW.andG.standfor |     |
| ---------------------------------------- | --- | --- | --- | ----- | ------ | --------------------- | --- |
| Figure5:Time-expandedRNNarchitecturewith |     |     |     | proc. | orig.  |                       |     |
WORD2VECandGLOVErespectively.
anappendedlayerinvolvinglinearbias.
Experiment Weconstructamodifiedneuralnetworkbyexplicitlyaddinga“postprocessingunit"
asthefirstlayeroftheRNNarchitecture(asinFigure5,wheretheappendedlayerisusedtotestthe
firststep(i.e.,removethemeanvector))ofthepostprocessingalgorithm.
In the modified neural network, the input word vectors are now v(w)−b instead of v(w). Here
b is a bias vector trained jointly with the rest of the neural network parameters. Note that this is
only a relabeling of the parameters from the perspective of the RNN architecture: the nonlinear
activationfunctionofthenodeisnowoperatedonA(v(w)−b)+b(cid:48) =Av(w)+(b(cid:48)−Ab)instead
ofthepreviousAv(w)+b(cid:48).
|     |     | Letb | proc. andb orig. | betheinferredbiaseswhenusingtheprocessedand |     |     |     |
| --- | --- | ---- | ---------------- | ------------------------------------------- | --- | --- | --- |
originalwordrepresentations,respectively.
Weitemizethecosinesimilaritybetweenb +µandb inTable6forthe5differentdatasets
|     |     |     | proc. |     | orig. |     |     |
| --- | --- | --- | ----- | --- | ----- | --- | --- |
and3differentneuralnetworkarchitectures. Ineachcase,thecosinesimilarityisremarkablylarge
(onaverage0.66,in300dimensions)–inotherwords,trainedneuralnetworksimplicitlypostprocess
thewordvectorsnearlyexactlyasweproposed. Thisagendaissuccessfullyimplementedinthe
contextofverifyingtheremovalofthemeanvector.
Thesecondstepofourpostprocessingalgorithm(i.e.,nullingawayfromthetopprincipalcomponents)
(cid:80)D
isequivalenttoapplyingaprojectionmatrixP =I− u u(cid:62)onthewordvectors,whereu is
|     |     |     |     |     | i=1 i i |     | i   |
| --- | --- | --- | --- | --- | ------- | --- | --- |
thei-thprincipalcomponentandDisthenumberoftheremoveddirections. Acomparativeanalysis
effortforthesecondstep(nullingthedominantPCAdirections)isthefollowing. Insteadofapplying
a bias term b, we multiply by a matrix Q to simulate the projection operation. The input word
vectorsarenowQ orig. v(w)insteadofv(w)fortheoriginalwordvectors,andQ proc. Pv(w)instead
ofPv(w)fortheprocessedvectors. TestingthesimilaritybetweenQ P andQ ,allowsusto
|     |     |     |     |     |     | orig. | proc. |
| --- | --- | --- | --- | --- | --- | ----- | ----- |
verifyiftheneuralnetworklearnstoconducttheprojectionoperationasproposed.
Inourexperiment,wefoundthatsucharesultcannotbeinferred. Onepossibilityisthattherearetoo
manyparametersinbothQ proc. andQ orig. ,whichaddsrandomnesstotheexperiment. Alternatively,
theneuralnetworkweightsmaynotbeabletolearnthesecondstepofthepostprocessingoperation
(indeed,inourexperimentspostprocessingsignificantlyboostedend-performanceofneuralnetwork
architectures).Amorecarefulexperimentalsetuptotestwhetherthesecondstepofthepostprocessing
operationislearntisleftasafutureresearchdirection.
15

PublishedasaconferencepaperatICLR2018
| C EXPERIMENTS |     | VARIOUS REPRESENTATIONS |     |     |     |
| ------------- | --- | ----------------------- | --- | --- | --- |
ON
In the main text, we have reported empirical results for two published word representations:
WORD2VECandGLOVE,eachin300dimensions. Inthissection,wereporttheresultsofthesame
experimentsinavarietyofothersettingstoshowthegeneralizationcapabilityofthepostprocessing
operation: representationstrainedviaWORD2VECandGLOVEalgorithmsindimensionsother
than300,otherrepresentationsalgorithms(specificallyTSCCAandRAND-WALK)andinmultiple
languages.
C.1 STATISTICSOFMULTILINGUALWORDREPRESENTATIONS
We use the publicly available TSCCA representations (Dhillon et al., 2012) on German, French,
Spanish,Italian,DutchandChinese. ThedetailedstatisticscanbefoundinTable8andthedecayof
theirsingularvaluesareplottedinFigure7.
|     | Language | Corpus | dim vocabsize | avg. (cid:107)v(w)(cid:107) | (cid:107)µ(cid:107) |
| --- | -------- | ------ | ------------- | --------------------------- | ------------------- |
2 2
| TSCCA-En | English                                               | Gigawords     | 200 300,000 | 4.38 | 0.78 |
| -------- | ----------------------------------------------------- | ------------- | ----------- | ---- | ---- |
| TSCCA-De | German                                                | Newswire      | 200 300,000 | 4.52 | 0.79 |
| TSCCA-Fr | French                                                | Gigaword      | 200 300,000 | 4.34 | 0.81 |
| TSCCA-Es | Spanish                                               | Gigaword      | 200 300,000 | 4.17 | 0.79 |
| TSCCA-It | Italian                                               | Newswire+Wiki | 200 300,000 | 4.34 | 0.79 |
| TSCCA-Nl | Dutch                                                 | Newswire+Wiki | 200 300,000 | 4.46 | 0.72 |
| TSCCA-Zh | Chinese                                               | Gigaword      | 200 300,000 | 4.51 | 0.89 |
| Table8:  | AdetaileddescriptionfortheTSCCAembeddingsinthispaper. |               |             |      |      |
0.07
TSCCA-De
0.06
TSCCA-En
|     |     | oitar ecnairav 0.05 | TSCCA-Es |     |     |
| --- | --- | ------------------- | -------- | --- | --- |
TSCCA-Fr
|     |     | 0.04 | TSCCA-It |     |     |
| --- | --- | ---- | -------- | --- | --- |
|     |     | 0.03 | TSCCA-Nl |     |     |
TSCCA-Zh
0.02
0.01
0.00
|     |     | 100 101 | 102 103 |     |     |
| --- | --- | ------- | ------- | --- | --- |
index
Figure7: Thedecayofthenormalizedsingularvaluesofmultilingualwordrepresentation.
C.2 MULTILINGUALGENERALIZATION
Inthissection,weperformthewordsimilaritytaskwiththeoriginalandtheprocessedTSCCAword
representationsinGermanandSpanishonthreeGermansimilaritydatasets(GUR65–aGerman
versionoftheRG65dataset,GUR350,andZG222intermsofrelatedness)(Zesch&Gurevych,2006)
andtheSpanishversionofRG65dataset(Camacho-Colladosetal.,2015). ThechoiceofDis2for
bothGermanandSpanish.
TheexperimentsetupandthesimilarityscoringalgorithmarethesameasthoseinSection3. The
detailedexperimentresultsareprovidedinTable9,fromwhichweobservethattheprocessedrepre-
sentationsareconsistentlybetterthantheoriginalones. Thisprovidesevidencetothegeneralization
capabilitiesofthepostprocessingoperationtomultiplelanguages(similaritydatasetsinSpanishand
Germanweretheonlyoneswecouldlocate).
C.3 GENERALIZATIONTODIFFERENTREPRESENTATIONALGORITHMS
Given the popularity and widespread use of WORD2VEC (Mikolov et al., 2013) and GLOVE
(Pennington et al., 2014), the main text has solely focused on their published publicly avalable
16

PublishedasaconferencepaperatICLR2018
TSCCA
language
|        | orig.         | proc. |
| ------ | ------------- | ----- |
| RG65   | Spanish 60.33 | 60.37 |
| GUR65  | German 61.75  | 64.39 |
| GUR350 | German 44.91  | 46.59 |
| ZG222  | German 30.37  | 32.92 |
Table9: Before-Afterresults(x100)onthewordsimilaritytaskinmultiplelanguages.
300-dimensionrepresentations. Inthissection,weshowthattheproposedpostprocessingalgorithm
generalizestootherrepresentationmethods. Specifically,wedemonstratethisonRAND-WALK
(obtainedviapersonalcommunication)andTSCCA(publiclyavailable)onalltheexperimentsof
Section3. ThechoiceofDis2forbothRAND-WALKandTSCCA.
Insummary,theperformanceimprovementsonthesimilaritytask,theconceptcategorizationtask,
theanalogytask,andthesemantictextualsimilaritydatasetareonaverage2.23%,2.39%,0.11%and
0.61%,respectively. ThedetailedstatisticsareprovidedinTable10,Table11,Table12andTable13,
respectively. Theseresultsareatestamenttothegeneralizationcapabilitiesofthepostprocessing
algorithmtootherrepresentationalgorithms.
| RAND-WALK     |             | TSCCA |
| ------------- | ----------- | ----- |
| orig.         | proc. orig. | proc. |
| RG65 80.66    | 82.96 47.53 | 47.67 |
| WS 65.89      | 74.37 54.21 | 54.35 |
| RW 45.11      | 51.23 43.96 | 43.72 |
| MEN 73.56     | 77.22 65.48 | 65.62 |
| MTurk 64.35   | 66.11 59.65 | 60.03 |
| SimLex 34.05  | 36.55 34.86 | 34.91 |
| SimVerb 16.05 | 21.84 23.79 | 23.83 |
Table10: Before-Afterresults(x100)onthewordsimilaritytaskonsevendatasets.
| RAND-WALK    |             | TSCCA |
| ------------ | ----------- | ----- |
| orig.        | proc. orig. | proc. |
| ap 59.83     | 62.36 60.00 | 63.42 |
| esslli 72.73 | 72.73 68.18 | 70.45 |
| battig 75.73 | 81.82 70.73 | 70.73 |
Table11: Before-Afterresults(x100)onthecategorizationtask.
| RAND-WALK  | TSCCA       |       |
| ---------- | ----------- | ----- |
| orig.      | proc. orig. | proc. |
| syn. 60.39 | 60.48 37.72 | 37.80 |
| sem. 83.55 | 83.82 14.54 | 14.55 |
| all 70.50  | 70.67 27.30 | 27.35 |
Table12: Before-Afterresults(x100)onthewordanalogytask.
C.4 ROLEOFDIMENSIONS
Themaintexthasfocusedonthedimensionchoiceofd=300,duetoitspopularity. Inthissection
we explore the role of the dimension in terms of both choice of D and the performance of the
postprocessingoperation–wedothisbyusingskip-grammodelonthe2010snapshotofWikipedia
corpus(Al-Rfouetal.,2013)totrainwordrepresentations. Wefirstobservethatthetwophenomena
ofSection2continuetohold:
• FromTable14weobservethattheratiobetweenthenormofµandthenormaverageofall
v(w)spansfrom1/3to1/4;
17

PublishedasaconferencepaperatICLR2018
|     |     |     | RAND-WALK  |       |     | TSCCA       |     |
| --- | --- | --- | ---------- | ----- | --- | ----------- | --- |
|     |     |     | orig.      | proc. |     | orig. proc. |     |
|     |     |     | 2012 38.03 | 37.66 |     | 44.51 44.63 |     |
|     |     |     | 2013 37.47 | 36.85 |     | 43.21 42.74 |     |
|     |     |     | 2014 46.06 | 48.32 |     | 52.85 52.87 |     |
|     |     |     | 2015 47.82 | 51.76 |     | 56.22 56.14 |     |
|     |     |     | SICK 51.58 | 51.76 |     | 56.15 56.11 |     |
|     |     |     | all 43.48  | 44.67 |     | 50.01 50.23 |     |
Table13: Before-Afterresults(x100)onthesemantictextualsimilaritytasks.
|     | dim                         |     | 300 400   | 500  | 600  | 700 800   | 900 1000  |
| --- | --------------------------- | --- | --------- | ---- | ---- | --------- | --------- |
|     | avg. (cid:107)v(w)(cid:107) |     | 4.51 5.17 | 5.91 | 6.22 | 6.49 6.73 | 6.95 7.15 |
2
(cid:107)µ(cid:107)
|     | 2   |     | 1.74 1.76 | 1.77 | 1.78 | 1.79 1.80 | 1.81 1.83 |
| --- | --- | --- | --------- | ---- | ---- | --------- | --------- |
Table14: Statisticsonwordrepresentationofdimensions300,400,...,and1000usingtheskip-gram
model.
• FromFigure8weobservethatthedecayofthevarianceratiosσ isnearexponentialfor
i
smallvaluesofiandremainsroughlyconstantoverthelaterones.
0.025
300
|     |     |     | 0.020          |     |     | 400 |     |
| --- | --- | --- | -------------- | --- | --- | --- | --- |
|     |     |     | oitar ecnairav |     |     | 500 |     |
600
0.015
700
800
|     |     |     | 0.010 |     |     | 900 |     |
| --- | --- | --- | ----- | --- | --- | --- | --- |
1000
0.005
0.000
|     |     |     | 100 | 101 |     | 102 | 103 |
| --- | --- | --- | --- | --- | --- | --- | --- |
index
Figure8: Thedecayofthenormalizedsingularvaluesofwordrepresentations.
AruleofthumbchoiceofDisaroundd/100. Wevalidatethisclaimempiricallybyperformingthe
tasksinSection3onwordrepresentationsofhigherdimensions,rangingfrom300to1000,where
wesettheparameterD =d/100. Insummary,theperformanceimprovementonthefouritemized
tasksofSection3are2.27%,3.37%,0.01and1.92%respectively;thedetailedresultscanbefound
inTable15,Table16,Table17,andTable18. Again,notethattheimprovementforanalogytasksis
marginal. Theseexperimentalresultsjustifytherule-of-thumbsettingofD =d/100,althoughwe
emphasizethattheimprovementscanbefurtheraccentuatedbytuningthechoiceofDbasedonthe
specificsetting.
| D EXPERIMENTS |     |     | WORD | ANALOGY | TASK |     |     |
| ------------- | --- | --- | ---- | ------- | ---- | --- | --- |
ON
ThedetailedperformanceontheanalogytaskisprovidedinTable19.
| E EXPERIMENTS |     | ON  | SEMANTIC | TEXTUAL |     | SIMILARITY | TASK |
| ------------- | --- | --- | -------- | ------- | --- | ---------- | ---- |
ThedetailedperformanceonthesemantictextualsimilarityisprovidedinTable20.
18

PublishedasaconferencepaperatICLR2018
|     |     |     | 300 |     | 400 |     | 500 |     | 600 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
Dim
|     |         |          | orig. proc. |       | orig. proc. |       | orig. proc. |       | orig. proc. |
| --- | ------- | -------- | ----------- | ----- | ----------- | ----- | ----------- | ----- | ----------- |
|     | RG65    | 73.57    | 74.72       | 75.64 | 79.87       | 77.72 | 81.97       | 77.59 | 80.7        |
|     |         | WS 70.25 | 71.95       |       | 70.8 72.88  | 70.39 | 72.73       | 71.64 | 74.04       |
|     |         | RW 46.25 | 49.11       | 45.97 | 47.63       |       | 46.6 48.59  |       | 45.7 47.81  |
|     | MEN     | 75.66    | 77.59       | 76.07 | 77.89       |       | 75.9 78.15  | 75.88 | 78.15       |
|     | Mturk   | 75.66    | 77.59       | 67.68 | 68.11       | 66.89 | 68.25       |       | 67.6 67.87  |
|     | SimLex  | 34.02    | 36.19       | 35.17 | 37.1        | 35.73 | 37.65       | 35.76 | 38.04       |
|     | SimVerb | 22.22    | 24.98       | 22.91 | 25.32       | 23.03 | 25.82       | 23.35 | 25.97       |
|     |         |          | 700         |       | 800         |       | 900         |       | 1000        |
Dim
|     |          |                                                               | orig. proc. |       | orig. proc. |          | orig. proc. |       | orig. proc. |
| --- | -------- | ------------------------------------------------------------- | ----------- | ----- | ----------- | -------- | ----------- | ----- | ----------- |
|     | RG65     |                                                               | 77.3 81.07  | 77.52 | 81.07       | 79.75    | 82.34       | 78.18 | 79.07       |
|     |          | WS 70.31                                                      | 73.02       | 71.52 | 74.65       | 71.19    | 73.06       |       | 71.5 74.78  |
|     |          | RW 45.86                                                      | 48.4        | 44.96 |             | 49 44.44 | 49.22       |       | 44.5 49.03  |
|     | MEN      | 75.84                                                         | 78.21       | 75.84 | 77.96       | 76.16    | 78.35       | 76.72 | 78.1        |
|     | Mturk    | 67.47                                                         | 67.79       | 67.67 |             | 68 67.98 | 68.87       | 68.34 | 69.44       |
|     | SimLex   |                                                               | 35.3 37.59  | 36.54 | 37.85       | 36.62    | 38.44       | 36.67 | 38.58       |
|     | SimVerb  | 22.81                                                         | 25.6        | 23.48 | 25.57       | 23.68    | 25.76       | 23.24 | 26.58       |
|     | Table15: | Before-Afterresults(x100)onwordsimilaritytaskonsevendatasets. |             |       |             |          |             |       |             |
|     |          |                                                               | 300         |       | 400         |          | 500         |       | 600         |
Dim
|     |        | orig.   | proc. | orig. | proc. | orig. | proc. | orig. | proc. |
| --- | ------ | ------- | ----- | ----- | ----- | ----- | ----- | ----- | ----- |
|     |        | ap 46.1 | 48.61 | 42.57 | 45.34 | 46.85 | 50.88 | 40.3  | 45.84 |
|     | esslli | 68.18   | 72.73 | 64.2  | 82.72 | 64.2  | 65.43 | 65.91 | 72.73 |
|     | battig | 71.6    | 77.78 | 68.18 | 75    | 68.18 | 70.45 | 46.91 | 66.67 |
|     |        |         | 700   |       | 800   |       | 900   |       | 1000  |
Dim
|     |        | orig.    | proc.                                             | orig. | proc. | orig. | proc. | orig. | proc. |
| --- | ------ | -------- | ------------------------------------------------- | ----- | ----- | ----- | ----- | ----- | ----- |
|     |        | ap 38.04 | 41.31                                             | 34.76 | 39.8  | 34.76 | 27.46 | 27.96 | 28.21 |
|     | esslli | 54.55    | 54.55                                             | 68.18 | 56.82 | 72.73 | 72.73 | 52.27 | 52.27 |
|     | battig | 62.96    | 66.67                                             | 67.9  | 69.14 | 49.38 | 59.26 | 51.85 | 46.91 |
|     |        | Table16: | Before-Afterresults(x100)onthecategorizationtask. |       |       |       |       |       |       |
|     |        |          | 300                                               |       | 400   |       | 500   |       | 600   |
Dim
|     |      | orig. | proc. | orig. | proc. | orig. | proc. | orig. | proc. |
| --- | ---- | ----- | ----- | ----- | ----- | ----- | ----- | ----- | ----- |
|     | syn. | 60.48 | 60.52 | 61.61 | 61.45 | 60.93 | 60.84 | 61.66 | 61.57 |
|     | sem. | 74.51 | 74.54 | 77.11 | 77.36 | 76.39 | 76.89 | 77.28 | 77.61 |
|     | all. | 66.86 | 66.87 | 68.66 | 68.69 | 67.88 | 68.11 | 68.77 | 68.81 |
|     |      |       | 700   |       | 800   |       | 900   |       | 1000  |
Dim
|              |      | orig.    | proc.                                          | orig. | proc. | orig.    | proc. | orig. | proc. |
| ------------ | ---- | -------- | ---------------------------------------------- | ----- | ----- | -------- | ----- | ----- | ----- |
|              | syn. | 60.94    | 61.02                                          | 68.38 | 68.34 | 60.47    | 60.30 | 67.56 | 67.30 |
|              | sem. | 77.24    | 77.26                                          | 77.24 | 77.35 | 76.76    | 76.90 | 76.71 | 76.51 |
|              | all. | 68.36    | 68.41                                          | 68.38 | 68.50 | 67.91    | 67.67 | 67.56 | 67.30 |
|              |      | Table17: | Before-Afterresults(x100)onthewordanalogytask. |       |       |          |       |       |       |
| F STATISTICS |      | OF       | TEXT CLASSIFICATION                            |       |       | DATASETS |       |       |       |
Weevaluatethewordrepresentations(withandwithoutpostprocessing)usingfourdifferentneural
networkarchitectures(CNN,vanilla-RNN,GRU-RNNandLSTM-RNN)onfivebenchmarks:
• themoviereview(MR)dataset(Pang&Lee,2005)whereeachreviewiscomposedbyonly
onesentence;
• thesubjectivity(SUBJ)dataset(Pang&Lee,2004)wherethealgorithmneedstodecide
whetherasentenceissubjectiveorobjective;
19

PublishedasaconferencepaperatICLR2018
300 400 500 600
Dim
orig. proc. orig. proc. orig. proc. orig. proc.
2012 54.51 54.95 54.31 54.57 55.13 56.23 55.35 56.03
2013 56.58 57.89 56.35 57.35 57.55 59.38 57.43 59.00
2014 59.6 61.92 59.57 61.62 61.19 64.38 61.10 63.86
2015 59.65 61.48 59.69 61.19 61.63 64.77 61.42 64.04
SICK 68.89 70.79 60.6 70.27 68.63 71.00 68.58 70.57
all 58.32 59.91 58.25 59.55 59.61 62.02 59.57 61.55
700 800 900 1000
Dim
orig. proc. orig. proc. orig. proc. orig. proc.
2012 55.52 56.49 54.47 54.85 54.69 55.18 54.34 54.78
2013 57.61 59.31 56.75 57.62 56.98 58.26 56.78 57.73
2014 61.57 64.77 60.51 62.83 60.89 63.34 60.78 63.03
2015 62.05 65.45 60.74 62.84 61.09 63.48 60.92 63.03
SICK 68.38 70.63 67.94 69.59 67.86 69.5 67.58 69.16
all 59.96 62.34 58.87 60.39 59.16 60.88 58.94 60.48
Table18: Before-Afterresults(x100)onthesemantictextualsimilaritytasks.
WORD2VEC GLOVE
orig. proc. orig. proc.
capital-common-countries 82.01 83.60 95.06 95.96
capital-world 78.38 80.08 91.89 92.31
city-in-state 69.56 69.88 69.56 70.45
currency 32.43 32.92 21.59 21.36
family 84.98 84.59 95.84 95.65
gram1-adjective-to-adverb 28.02 27.72 40.42 39.21
gram2-opposite 40.14 40.51 31.65 30.91
gram3-comparative 89.19 89.26 86.93 87.09
gram4-superlative 82.71 83.33 90.46 90.59
gram5-present-participle 79.36 79.64 82.95 82.76
gram6-nationality-adjective 90.24 90.36 90.24 90.24
gram7-past-tense 66.03 66.53 63.91 64.87
gram8-plural 91.07 90.61 95.27 95.36
gram9-plural-verbs 68.74 67.58 67.24 68.05
Table19: Before-Afterresults(x100)onthewordanalogytask.
• theTRECquestiondataset(Li&Roth,2002)whereallthequestionsinthisdatasethasto
bepartitionedintosixcategories;
• theIMDbdataset(Maasetal.,2011)–eachreviewconsistsofseveralsentences;
• theStanfordsentimenttreebank(SST)dataset(Socheretal.,2013a),whereweonlyusethe
fullsentencesasthetrainingdata.
InTREC,SSTandIMDb,thedatasetshavealreadybeensplitintotrain/testsets. Otherwiseweuse
10-foldcrossvalidationintheremainingdatasets(i.e.,MRandSUBJ).Detailedstatisticsofvarious
featuresofeachofthedatasetsareprovidedinTable21.
G PROOF OF THEOREM A.1
Given the similarity between the setup in Theorem 2.2 in (Arora et al., 2016) and Theorem A.1,
manypartsoftheoriginalproofcanbereusedexceptonekeyaspect–theconcentrationofZ(c). We
summarizethispartinthefollowinglemma:
20

PublishedasaconferencepaperatICLR2018
WORD2VEC GLOVE
orig. proc. orig. proc.
2012.MSRpar 42.12 43.85 44.54 44.09
2012.MSRvid 72.07 72.16 64.47 68.05
2012.OnWN 69.38 69.48 53.07 65.67
2012.SMTeuroparl 53.15 54.32 41.74 45.28
2012.SMTnews 49.37 48.53 37.54 47.22
2013.FNWN 40.70 41.96 37.54 39.34
2013.OnWN 67.87 68.17 47.22 58.60
2013.headlines 61.88 63.81 49.73 57.20
2014.OnWN 74.61 74.78 57.41 67.56
2014.deft-forum 32.19 33.26 21.55 29.39
2014.deft-news 66.83 65.96 65.14 71.45
2014.headlines 58.01 59.58 47.05 52.60
2014.images 73.75 74.17 57.22 68.28
2014.tweet-news 71.92 72.07 58.32 66.13
2015.answers-forum 46.35 46.80 30.02 39.86
2015.answers-students 68.07 67.99 49.20 62.38
2015.belief 59.72 60.42 44.05 57.68
2015.headlines 61.47 63.45 46.22 53.31
2015.images 78.09 78.08 66.63 73.20
SICK 70.10 70.20 65.14 67.85
Table20: Before-Afterresults(x100)onthesemantictextualsimilaritytasks.
c l Train Test
MR 2 20 10,662 10-foldcrossvalidation
SUBJ 2 23 10,000 10-foldcrossvalidation
TREC 6 10 5,952 500
SST 5 18 11,855 2,210
IMDb 2 100 25,000 25,000
Table 21: Statistics for the five datasets after tokenization: c represents the number of classes; l
representstheaveragesentencelength;Trainrepresentsthesizeofthetrainingset;andTestrepresent
thesizeofthetestset.
LemmaG.1 Letcbearandomvariableuniformlydistributedovertheunitsphere,weprovethat
withhighprobability,Z(c)/|V|convergestoaconstantZ:
p((1−(cid:15) )Z ≤Z(c)≤(1+(cid:15) )Z)≥1−δ,
z z
where(cid:15) =Ω((D+1)/|V|)andδ =Ω((DA2+(cid:107)µ(cid:107)2)/d).
z
Ourproofdiffersfromtheonein(Aroraetal.,2016)intwoways: (a)wetreatv(w)asdeterministic
parametersinsteadofrandomvariablesandprovetheLemmabyshowingacertainconcentrationof
measure;(b)theasymmetricpartsµandu ,...,u ,(whichdidnotexistintheoriginalproof),needto
1 D
becarefullyaddressedtocompletetheproof.
21

PublishedasaconferencepaperatICLR2018
G.1 PROOFOFLEMMAG.1
Giventheconstraintsonthewordvectors(3),thepartitionfunctionZ(c)canberewrittenas,
(cid:88)
|     | Z(c)= |     | exp(c(cid:62)v(w)) |     |     |     |     |     |     |     |
| --- | ----- | --- | ------------------ | --- | --- | --- | --- | --- | --- | --- |
v∈V
|     |     |          | (cid:32)        | (cid:32)     | D        |               |          | (cid:33)(cid:33)       |           |     |
| --- | --- | -------- | --------------- | ------------ | -------- | ------------- | -------- | ---------------------- | --------- | --- |
|     |     | (cid:88) |                 |              | (cid:88) |               |          |                        |           |     |
|     |     | =        | exp             | c(cid:62) µ+ | α        | (w)u          | +v˜(w)   |                        |           |     |
|     |     |          |                 |              |          | i             | i        |                        |           |     |
|     |     | v∈V      |                 |              | i=1      |               |          |                        |           |     |
|     |     |          |                 | (cid:34)     | D        |               | (cid:35) |                        |           |     |
|     |     | (cid:88) |                 |              | (cid:89) |               |          |                        |           |     |
|     |     | =        | exp(c(cid:62)µ) |              | exp(α    | (w)c(cid:62)u | ) exp    | (cid:0) c(cid:62)v˜(w) | (cid:1) . |     |
|     |     |          |                 |              |          | i             | i        |                        |           |     |
|     |     | v∈V      |                 |              | i=1      |               |          |                        |           |     |
Theequationabovesuggeststhatwecandividetheproofintofiveparts.
Step1: foreveryunitvectorc,onehas,
1 (cid:88)
|     |     |     | (cid:0) c(cid:62)v˜(w) | (cid:1) | →E (cid:0) | (cid:0) c(cid:62)v˜ | (cid:1)(cid:1) |          |     |     |
| --- | --- | --- | ---------------------- | ------- | ---------- | ------------------- | -------------- | -------- | --- | --- |
|     |     | exp |                        |         | f          | exp                 | ,              | as|V|→∞. |     | (5) |
|V|
w∈V
| Proof LetM,N | beapositiveinteger,andletA |     |     |     |     | ⊂Rdsuchthat, |     |     |     |     |
| ------------ | -------------------------- | --- | --- | --- | --- | ------------ | --- | --- | --- | --- |
M
|     |     |     | (cid:26) |     |      |                    |     | (cid:27) |     |     |
| --- | --- | --- | -------- | --- | ---- | ------------------ | --- | -------- | --- | --- |
|     |     |     |          |     | M −1 |                    |     | M        |     |     |
|     |     | A   | = v˜∈Rd  |     | :    | <exp(c(cid:62)v˜)≤ |     |          | .   |     |
M,N
|     |     |     |     |     | N   |     |     | N   |     |     |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
SinceA canberepresentedbyaunionofcountabledisjointrectangles,weknowthatforevery
M,N
| M,N ∈N , |     |     |     |     |     |     |     |     |     |     |
| -------- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
+
(cid:90)
1 (cid:88)
|     |     |     | 1(v˜(w)∈A |     |     | )=  | f(v˜)dv˜. |     |     |     |
| --- | --- | --- | --------- | --- | --- | --- | --------- | --- | --- | --- |
|     |     | |V| |           |     | M,N |     |           |     |     |     |
AM,N
w∈V
| Further,sinceA |          | aredisjointfordifferentM’sandRd |         |          |          |           | =∪∞ | A                    | ,onehas, |     |
| -------------- | -------- | ------------------------------- | ------- | -------- | -------- | --------- | --- | -------------------- | -------- | --- |
|                | M,N      |                                 |         |          |          |           | M=1 | M,N                  |          |     |
|                | 1        |                                 |         | ∞        | 1        |           |     |                      |          |     |
|                | (cid:88) | (cid:0) c(cid:62)v˜(w)          | (cid:1) | (cid:88) | (cid:88) |           |     | )exp(c(cid:62)v˜(w)) |          |     |
|                |          | exp                             |         | =        |          | 1(v˜(w)∈A |     | M,N                  |          |     |
|                | |V|      |                                 |         |          | |V|      |           |     |                      |          |     |
|                | w∈V      |                                 |         | M=1      | w∈V      |           |     |                      |          |     |
∞
|     |     |     |     | (cid:88) | 1 (cid:88) |           |     | M   |     |     |
| --- | --- | --- | --- | -------- | ---------- | --------- | --- | --- | --- | --- |
|     |     |     |     | ≤        |            | 1(v˜(w)∈A |     | )   |     |     |
M,N
|     |     |     |     |          | |V|      |           |     | N   |     |     |
| --- | --- | --- | --- | -------- | -------- | --------- | --- | --- | --- | --- |
|     |     |     |     | M=1      | w∈V      |           |     |     |     |     |
|     |     |     |     | ∞        | (cid:90) |           |     |     |     |     |
|     |     |     |     | (cid:88) | M        |           |     |     |     |     |
|     |     |     |     | →        |          | f(v˜)dv˜. |     |     |     |     |
N AM,N
M=1
TheabovestatementholdsforeveryN. LetN →∞,bydefinitionofintegration,onehas,
|     |     |     | ∞          | (cid:90) |            |     |         |                        |     |     |
| --- | --- | --- | ---------- | -------- | ---------- | --- | ------- | ---------------------- | --- | --- |
|     |     |     | (cid:88) M |          |            |     | (cid:0) | (cid:0) (cid:1)(cid:1) |     |     |
|     |     | lim |            |          | f(v˜)dv˜=E |     | exp     | c(cid:62)v˜            | ,   |     |
|     |     |     | N          |          |            |     | f       |                        |     |     |
|     |     | N→∞ |            | AM,N     |            |     |         |                        |     |     |
M=1
whichyields,
1 (cid:88)
|     |     | exp | (cid:0) c(cid:62)v˜(w) | (cid:1) | ≤E (cid:0) | exp (cid:0) c(cid:62)v˜ | (cid:1)(cid:1) , | as|V|→∞. |     | (6) |
| --- | --- | --- | ---------------------- | ------- | ---------- | ----------------------- | ---------------- | -------- | --- | --- |
f
|V|
w∈V
Similarly,onehas,
∞
|     |     | 1 (cid:88) |                        |         |            | (cid:88)            | M −1(cid:90)   |          |     |     |
| --- | --- | ---------- | ---------------------- | ------- | ---------- | ------------------- | -------------- | -------- | --- | --- |
|     |     | exp        | (cid:0) c(cid:62)v˜(w) | (cid:1) | ≥ lim      |                     |                | f(v˜)dv˜ |     |     |
|     |     | |V|        |                        |         | N→∞        |                     | N              |          |     |     |
|     |     | w∈V        |                        |         |            | M=1                 |                | AM,N     |     |     |
|     |     |            |                        |         | =E (cid:0) | (cid:0) c(cid:62)v˜ | (cid:1)(cid:1) |          |     |     |
|     |     |            |                        |         | exp        |                     | , as|V|→∞.     |          |     | (7) |
f
Putting(6)and(7)proves(5).
22

PublishedasaconferencepaperatICLR2018
| theexpectedvalue,E |     | (cid:0) | (cid:0)         | (cid:1)(cid:1)             |     |     |     |     |     |
| ------------------ | --- | ------- | --------------- | -------------------------- | --- | --- | --- | --- | --- |
| Step2:             |     |         | exp c(cid:62)v˜ | isaconstantindependentofc: |     |     |     |     |     |
f
|     |     |     | E   | (cid:0) exp (cid:0) c(cid:62)v˜ | (cid:1)(cid:1) =Z | .   |     |     | (8) |
| --- | --- | --- | --- | ------------------------------- | ----------------- | --- | --- | --- | --- |
|     |     |     | f   |                                 |                   | 0   |     |     |     |
Proof LetQ ∈ Rd×d beaorthonormalmatrixsuchthatQ(cid:62)c = cwherec = (1,0,...,0)(cid:62) and
|     |     |     |     |     |     | 0   |     | 0   |     |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
det(Q)=1,thenwehavef(v˜)=f(Qv˜),and,
(cid:90)
|     | E (cid:0) | (cid:0) (cid:1)(cid:1) |            | (cid:0)     | (cid:1) |     |     |     |     |
| --- | --------- | ---------------------- | ---------- | ----------- | ------- | --- | --- | --- | --- |
|     | exp       | c(cid:62)v˜            | = f(v˜)exp | c(cid:62)v˜ | dv˜     |     |     |     |     |
|     | f         | 0                      |            |             | 0       |     |     |     |     |
v˜
(cid:90)
|     |     |     |             |     | (cid:0) (cid:1) |           |     |     |     |
| --- | --- | --- | ----------- | --- | --------------- | --------- | --- | --- | --- |
|     |     |     | = f(Qv˜)exp |     | c(cid:62)Qv˜    | det(Q)dv˜ |     |     |     |
v˜
(cid:90)
|     |     |     | f(v˜(cid:48))exp | (cid:0) | c(cid:62)v˜(cid:48)(cid:1) dv˜(cid:48) | =E  | (cid:0) (cid:0) c(cid:62)v˜ | (cid:1)(cid:1) |     |
| --- | --- | --- | ---------------- | ------- | -------------------------------------- | --- | --------------------------- | -------------- | --- |
|     |     |     | =                |         |                                        |     | f exp                       | ,              |     |
v˜(cid:48)
whichproves(8).
Step3: foranyvectorµ,onehasthefollowingconcentrationproperty,
|     |                            |         |         | (cid:20) | (cid:18) 1 (cid:19) | (cid:107)µ(cid:107)2 | 1   | (cid:21) |     |
| --- | -------------------------- | ------- | ------- | -------- | ------------------- | -------------------- | --- | -------- | --- |
|     | (cid:0) (cid:0) c(cid:62)µ | (cid:1) | (cid:1) |          |                     |                      |     |          |     |
| p   | |exp                       | −1|>k   |         | ≤2 exp   | −                   | +                    |     |          | (9) |
4 d−1log2(1−k)
√
Proof Letc ,...,c bei.i.d.N(0,1),andletC = (cid:80)d c2,thenc=(c ,...,c )/ C isuniformover
| 1   | d   |     |     |     | i=1 i |     | 1 d |     |     |
| --- | --- | --- | --- | --- | ----- | --- | --- | --- | --- |
unitsphere. Sincecisuniform,thenwithoutlossofgeneralitywecanconsiderµ=((cid:107)µ(cid:107),0,...,0).
|                          |     | (cid:16)             | √   | (cid:17)                                 |     |     |     |     |     |
| ------------------------ | --- | -------------------- | --- | ---------------------------------------- | --- | --- | --- | --- | --- |
| Thusitsufficestoboundexp |     | (cid:107)µ(cid:107)c | / C |                                          |     |     |     |     |     |
|                          |     |                      | 1   | . Wedividetheproofintothefollowingsteps: |     |     |     |     |     |
• C followschi-squaredistributionwiththedegreeoffreedomofd,thusC canbebounded
by(Laurent&Massart,2000),
√
|     |     | p(C | ≥d+2 | dx+2x)≤exp(−x),∀x>0. |     |     |     |     | (10) |
| --- | --- | --- | ---- | -------------------- | --- | --- | --- | --- | ---- |
√
|     |     |     | p(C | ≤d−2 | dx)≤exp(−x),∀x>0. |     |     |     | (11) |
| --- | --- | --- | --- | ---- | ----------------- | --- | --- | --- | ---- |
• Thereforeforanyx>0,onehas,
√
|     |     |     | (cid:16)  |     | (cid:17) |          |     |     |     |
| --- | --- | --- | --------- | --- | -------- | -------- | --- | --- | --- |
|     |     |     | p |C−d|≥2 |     | dx       | ≤exp(−x) |     |     |     |
Letx=1/4d,onehas,
|     |     |     |     |           |     | (cid:18) 1 | (cid:19) |     |     |
| --- | --- | --- | --- | --------- | --- | ---------- | -------- | --- | --- |
|     |     |     | p(C | >d+1)≤exp |     | −          | ,        |     |     |
4d
|     |     |     |     |     |     | (cid:18) | (cid:19) |     |     |
| --- | --- | --- | --- | --- | --- | -------- | -------- | --- | --- |
1
|     |     |     | p(C | <d−1)≤exp |     | −   | .   |     |     |
| --- | --- | --- | --- | --------- | --- | --- | --- | --- | --- |
4d
• Sincec isaGaussianrandomvariablewithvariance1,byChebyshev’sinequality,onehas,
1
|     |     |      |      | y2  |      |       | y2  |     |     |
| --- | --- | ---- | ---- | --- | ---- | ----- | --- | --- | --- |
|     |     | p(yc | ≥k)≤ | ,   | p(yc | ≤−k)≤ | ,∀k | >0  |     |
|     |     | i    |      |     | i    |       |     |     |     |
|     |     |      |      | k2  |      |       | k2  |     |     |
andthereforethus,
y2
|     |     | p(exp(yc |     | i )−1>k)≤ |     |     | ,   |     |     |
| --- | --- | -------- | --- | --------- | --- | --- | --- | --- | --- |
log2(1+k)
y2
|     |     | p(exp(yc | i )−1<−k)≤ |     |     |     | , ∀k >0. |     |     |
| --- | --- | -------- | ---------- | --- | --- | --- | -------- | --- | --- |
log(1−k)2
23

PublishedasaconferencepaperatICLR2018
(cid:16) √ (cid:17)
• Thereforewecanboundexp (cid:107)µ(cid:107)c / C by,
1
(cid:18) (cid:18) (cid:19) (cid:19)
(cid:107)µ(cid:107)c
p exp √ 1 −1>k ≤p(C >d+1)
C
(cid:18) (cid:18) (cid:19) (cid:12) (cid:19)
+p exp (cid:107) √ µ(cid:107)c 1 −1>k (cid:12) (cid:12)C <d+1 p(C <d+1)
C (cid:12)
(cid:18) (cid:19) (cid:18) (cid:18) (cid:19) (cid:19)
1 (cid:107)µ(cid:107)c
≤exp − +p exp √ 1 −1>k
4d d+1
(cid:18) 1 (cid:19) (cid:107)µ(cid:107)2 1
=exp − + .
4d d+1log(1−k)2
(cid:18) (cid:18) (cid:107)µ(cid:107)c (cid:19) (cid:19) (cid:18) 1 (cid:19) (cid:107)µ(cid:107)2 1
p exp √ 1 −1<−k ≤exp − + .
C 4d d−1log2(1+k)
Combiningthetwoinequalitiesabove,onehas(9)proved.
Step4: WearenowreadytoproveconvergenceofZ(c). With(9),letC ⊂Rdsuchthat,
C = (cid:8) c: (cid:12) (cid:12)exp(c(cid:62)µ)−1 (cid:12) (cid:12)<k, (cid:12) (cid:12)exp(Ac(cid:62)u
i
)−1 (cid:12) (cid:12)<k, (cid:12) (cid:12)exp(−Ac(cid:62)u
i
)−1 (cid:12) (cid:12)<k∀i=1,...,D (cid:9)
ThenwecanboundtheprobabilityonC by,
D
p(C)≥p (cid:0)(cid:12) (cid:12)exp(c(cid:62)µ)−1 (cid:12) (cid:12)<k (cid:1) + (cid:88) p( (cid:12) (cid:12)exp(Ac(cid:62)u
i
)−1 (cid:12) (cid:12)<k)−2D
i=1
(cid:18) 1 (cid:19) 2DA2 1 (cid:107)µ(cid:107)2 1
≥1−(2D+1)exp − − − .
4d d−1 log2(1−k) d−1log2(1−k)
Next,weneedtoshowthatforeveryw,thecorrespondingC(w),i.e.,
C(w)= (cid:8) c: (cid:12) (cid:12)exp(c(cid:62)µ)−1 (cid:12) (cid:12)<k, (cid:12) (cid:12)exp(α
i
(w)c(cid:62)u
i
)−1 (cid:12) (cid:12)<k, ∀i=1,...,D (cid:9)
Weobservethatα (w)isboundedbyA,thereforeforanycthat,
i
min(exp(−Ac(cid:62)u ),exp(Ac(cid:62)u ))≤exp(α c(cid:62)u )≤max(exp(−Ac(cid:62)u ),exp(Ac(cid:62)u )),
i i i i i i
andthus,
min(exp(−Ac(cid:62)u ),exp(Ac(cid:62)u ))−1≤exp(α c(cid:62)u )−1≤max(exp(−Ac(cid:62)u ),exp(Ac(cid:62)u ))−1,
i i i i i i
whichyields,
|exp(α c(cid:62)u )−1|≤max(|exp(−Ac(cid:62)u )−1|,|exp(Ac(cid:62)u )−1|)<k.
i i i i
ThereforeweproveC(w)⊃C. Assemblingeverythingtogether,onehas,
(cid:32)(cid:12) (cid:12) (cid:89) D (cid:12) (cid:12) (cid:33)
p (cid:12)exp(c(cid:62)µ) exp(α (w)c(cid:62)u )−1(cid:12)>(D+1)k, ∀i=1,...,D,∀w ∈V
(cid:12) i i (cid:12)
(cid:12) (cid:12)
i=1
≤p(C¯)
1 2DA2 1 (cid:107)µ(cid:107)2 1
≤(2D+1)exp(− )+ +
4d d−1 log2(1−k) d−1log2(1−k)
Foreveryc∈C,onehas,
1 (D+1)k
|Z(c)−Z |≤ Z .
|V| 0 |V| 0
LetZ =|V|Z ,onecanconcludethat,
0
p((1−(cid:15) )Z ≤Z(c)≤(1+(cid:15) )Z)≥1−δ,
z z
where(cid:15) =Ω((D+1)/|V|)andδ =Ω(DA2/d).
z
24

PublishedasaconferencepaperatICLR2018
G.2 PROOFOFTHEOREMA.1
HavingLemmaG.1ready,wecanfollowthesameproofasin(Aroraetal.,2016)thatbothp(w)and
p(w,w(cid:48))arecorrelatedwith(cid:107)v(w)(cid:107),formally
(cid:107)v(w)(cid:107)2
logp(w)→ −logZ, as|V|→∞, (12)
2d
(cid:107)v(w)+v(w(cid:48))(cid:107)2
logp(w,w(cid:48))→ −logZ, as|V|→∞. (13)
2d
Therefore,theinferencepresentedin(Aroraetal.,2016)(i.e.,(4))isobviousbyassembling(12)and
(13)together:
v(w)(cid:62)v(w(cid:48))
PMI(w,w(cid:48))→ , as|V|→∞.
d
25
