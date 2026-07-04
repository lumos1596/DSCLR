PublishedasaconferencepaperatICLR2018
WORD TRANSLATION WITHOUT PARALLEL DATA
AlexisConneau∗†‡,GuillaumeLample∗†§,
Marc’AurelioRanzato†,LudovicDenoyer§,Herve´ Je´gou†
{aconneau,glample,ranzato,rvj}@fb.com
ludovic.denoyer@upmc.fr
ABSTRACT
State-of-the-art methods for learning cross-lingual word embeddings have relied
onbilingualdictionariesorparallelcorpora. Recentstudiesshowedthattheneed
for parallel data supervision can be alleviated with character-level information.
While these methods showed encouraging results, they are not on par with their
supervised counterparts and are limited to pairs of languages sharing a common
alphabet. Inthiswork,weshowthatwecanbuildabilingualdictionarybetween
twolanguageswithoutusinganyparallelcorpora,byaligningmonolingualword
embeddingspacesinanunsupervisedway. Withoutusinganycharacterinforma-
tion, our model even outperforms existing supervised methods on cross-lingual
tasks for some language pairs. Our experiments demonstrate that our method
worksverywellalsofordistantlanguagepairs,likeEnglish-RussianorEnglish-
Chinese. WefinallydescribeexperimentsontheEnglish-Esperantolow-resource
languagepair,onwhichthereonlyexistsalimitedamountofparalleldata,toshow
thepotentialimpactofourmethodinfullyunsupervisedmachinetranslation. Our
code,embeddingsanddictionariesarepubliclyavailable1.
1 INTRODUCTION
Most successful methods for learning distributed representations of words (e.g. Mikolov et al.
(2013c;a); Pennington et al. (2014); Bojanowski et al. (2017)) rely on the distributional hypoth-
esis of Harris (1954), which states that words occurring in similar contexts tend to have similar
meanings. Levy & Goldberg (2014) show that the skip-gram with negative sampling method of
Mikolovetal.(2013c)amountstofactorizingaword-contextco-occurrencematrix, whoseentries
arethepointwisemutualinformationoftherespectivewordandcontextpairs. Exploitingwordco-
occurrencestatisticsleadstowordvectorsthatreflectthesemanticsimilaritiesanddissimilarities:
similarwordsarecloseintheembeddingspaceandconversely.
Mikolovetal.(2013b)firstnoticedthatcontinuouswordembeddingspacesexhibitsimilarstructures
acrosslanguages,evenwhenconsideringdistantlanguagepairslikeEnglishandVietnamese. They
proposedtoexploitthissimilaritybylearningalinearmappingfromasourcetoatargetembedding
space. They employed a parallel vocabulary of five thousand words as anchor points to learn this
mappingandevaluatedtheirapproachonawordtranslationtask. Sincethen,severalstudiesaimed
at improving these cross-lingual word embeddings (Faruqui & Dyer (2014); Xing et al. (2015);
Lazaridouetal.(2015);Ammaretal.(2016);Artetxeetal.(2016);Smithetal.(2017)),buttheyall
relyonbilingualwordlexicons.
Recentattemptsatreducingtheneedforbilingualsupervision(Smithetal.,2017)employidentical
characterstringstoformaparallelvocabulary. TheiterativemethodofArtetxeetal.(2017)gradu-
allyalignsembeddingspaces,startingfromaparallelvocabularyofaligneddigits. Thesemethods
arehoweverlimitedtosimilarlanguagessharingacommonalphabet,suchasEuropeanlanguages.
Somerecentmethodsexploreddistribution-basedapproach(Caoetal.,2016)oradversarialtraining
Zhangetal.(2017b)toobtaincross-lingualwordembeddingswithoutanyparalleldata.Whilethese
∗Equalcontribution.Orderhasbeendeterminedwithacoinflip.
†FacebookAIResearch
‡LIUM,UniversityofLeMans
§SorbonneUniversite´s,UPMCUnivParis06,UMR7606,LIP6
1https://github.com/facebookresearch/MUSE
1
8102
naJ
03
]LC.sc[
3v78040.0171:viXra

PublishedasaconferencepaperatICLR2018
approachessoundappealing,theirperformanceissignificantlybelowsupervisedmethods. Tosum
up, currentmethodshaveeithernotreachedcompetitiveperformance, ortheystillrequireparallel
data,suchasalignedcorpora(Gouwsetal.,2015;Vulic&Moens,2015)oraseedparallellexicon
(Duongetal.,2016).
Inthispaper,weintroduceamodelthateitherisonpar,oroutperformssupervisedstate-of-the-art
methods,withoutemployinganycross-lingualannotateddata. Weonlyusetwolargemonolingual
corpora,oneinthesourceandoneinthetargetlanguage. Ourmethodleveragesadversarialtraining
tolearnalinearmappingfromasourcetoatargetspaceandoperatesintwosteps. First,inatwo-
playergame, adiscriminatoristrainedtodistinguishbetweenthemappedsourceembeddingsand
thetargetembeddings,whilethemapping(whichcanbeseenasagenerator)isjointlytrainedtofool
the discriminator. Second, we extract a synthetic dictionary from the resulting shared embedding
spaceandfine-tunethemappingwiththeclosed-formProcrustessolutionfromScho¨nemann(1966).
Since the method is unsupervised, cross-lingual data can not be used to select the best model. To
overcomethisissue,weintroduceanunsupervisedselectionmetricthatishighlycorrelatedwiththe
mappingqualityandthatweusebothasastoppingcriterionandtoselectthebesthyper-parameters.
Insummary,thispapermakesthefollowingmaincontributions:
• We present an unsupervised approach that reaches or outperforms state-of-the-art super-
visedapproachesonseverallanguagepairsandonthreedifferentevaluationtasks,namely
word translation, sentence translation retrieval, and cross-lingual word similarity. On
a standard word translation retrieval benchmark, using 200k vocabularies, our method
reaches66.2%accuracyonEnglish-Italianwhilethebestsupervisedapproachisat63.7%.
• Weintroduceacross-domainsimilarityadaptationtomitigatetheso-calledhubnessprob-
lem(pointstendingtobenearestneighborsofmanypointsinhigh-dimensionalspaces). It
isinspiredbytheself-tuningmethodfromZelnik-manor&Perona(2005),butadaptedto
ourtwo-domainscenarioinwhichwemustconsiderabi-partitegraphforneighbors. This
approachsignificantlyimprovestheabsoluteperformance,andoutperformsthestateofthe
artbothinsupervisedandunsupervisedsetupsonword-translationbenchmarks.
• Weproposeanunsupervisedcriterionthatishighlycorrelatedwiththequalityofthemap-
ping,thatcanbeusedbothasastoppingcriterionandtoselectthebesthyper-parameters.
• Wereleasehigh-qualitydictionariesfor12orientedlanguagespairs, aswellasthecorre-
spondingsupervisedandunsupervisedwordembeddings.
• Wedemonstratetheeffectivenessofourmethodusinganexampleofalow-resourcelan-
guage pair where parallel corpora are not available (English-Esperanto) for which our
methodisparticularlysuited.
Thepaperisorganizedasfollows. Section2describesourunsupervisedapproachwithadversarial
training and our refinement procedure. We then present our training procedure with unsupervised
model selection in Section 3. We report in Section 4 our results on several cross-lingual tasks for
severallanguagepairsandcompareourapproachtosupervisedmethods. Finally, weexplainhow
ourapproachdiffersfromrecentrelatedworkonlearningcross-lingualwordembeddings.
2 MODEL
In this paper, we always assume that we have two sets of embeddings trained independently on
monolingualdata. Ourworkfocusesonlearningamappingbetweenthetwosetssuchthattransla-
tionsarecloseinthesharedspace. Mikolovetal.(2013b)showthattheycanexploitthesimilarities
of monolingual embedding spaces to learn such a mapping. For this purpose, they use a known
dictionaryofn = 5000pairsofwords{x ,y } ,andlearnalinearmappingW betweenthe
i i i∈{1,n}
sourceandthetargetspacesuchthat
W(cid:63) = argmin (cid:107)WX−Y(cid:107) (1)
F
W∈Md(R)
wheredisthedimensionoftheembeddings,M (R)isthespaceofd×dmatricesofrealnumbers,
d
andX andY aretwoalignedmatricesofsized×ncontainingtheembeddingsofthewordsinthe
parallelvocabulary. Thetranslationtofanysourcewordsisdefinedast=argmax cos(Wx ,y ).
t s t
2

PublishedasaconferencepaperatICLR2018
Figure1:Toyillustrationofthemethod.(A)Therearetwodistributionsofwordembeddings,Englishwords
| X   |     |     | Y,  |     |     |     |
| --- | --- | --- | --- | --- | --- | --- |
in red denoted by and Italian words in blue denoted by which we want to align/translate. Each dot
representsawordinthatspace.Thesizeofthedotisproportionaltothefrequencyofthewordsinthetraining
corpusofthatlanguage.(B)Usingadversariallearning,welearnarotationmatrixW whichroughlyalignsthe
twodistributions. Thegreenstarsarerandomlyselectedwordsthatarefedtothediscriminatortodetermine
whetherthetwowordembeddingscomefromthesamedistribution.(C)ThemappingW isfurtherrefinedvia
Procrustes. Thismethodusesfrequentwordsalignedbythepreviousstepasanchorpoints,andminimizesan
energyfunctionthatcorrespondstoaspringsystembetweenanchorpoints. Therefinedmappingisthenused
tomapallwordsinthedictionary. (D)Finally,wetranslatebyusingthemappingW andadistancemetric,
dubbed CSLS, that expands the space where there is high density of points (like the area around the word
“cat”),sothat“hubs”(liketheword“cat”)becomelessclosetootherwordvectorsthantheywouldotherwise
(comparetothesameregioninpanel(A)).
Inpractice,Mikolovetal.(2013b)obtainedbetterresultsonthewordtranslationtaskusingasim-
plelinearmapping,anddidnotobserveanyimprovementwhenusingmoreadvancedstrategieslike
multilayer neural networks. Xing et al. (2015) showed that these results are improved by enforc-
ing an orthogonality constraint on W. In that case, the equation (1) boils down to the Procrustes
problem, whichadvantageouslyoffersaclosedformsolutionobtainedfromthesingularvaluede-
composition(SVD)ofYXT:
| W(cid:63) | = argmin(cid:107)WX−Y(cid:107) |     | =UVT,withUΣVT |     | =SVD(YXT). |     |
| --------- | ------------------------------ | --- | ------------- | --- | ---------- | --- |
|           |                                | F   |               |     |            | (2) |
W∈Od(R)
Inthispaper,weshowhowtolearnthismappingW withoutcross-lingualsupervision;anillustration
of the approach is given in Fig. 1. First, we learn an initial proxy of W by using an adversarial
criterion. Then, weusethewordsthatmatchthebestasanchorpointsforProcrustes. Finally, we
improveperformanceoverlessfrequentwordsbychangingthemetricofthespace,whichleadsto
spreadmoreofthosepointsindenseregions. Next,wedescribethedetailsofeachofthesesteps.
2.1 DOMAIN-ADVERSARIALSETTING
In this section, we present our domain-adversarial approach for learning W without cross-lingual
supervision. LetX ={x ,...,x }andY ={y ,...,y }betwosetsofnandmwordembeddings
|     | 1 n |     | 1 m |     |     |     |
| --- | --- | --- | --- | --- | --- | --- |
comingfromasourceandatargetlanguagerespectively.Amodelistrainedtodiscriminatebetween
elementsrandomlysampledfromWX ={Wx ,...,Wx }andY. Wecallthismodelthediscrim-
|     |     |     | 1 n |     |     |     |
| --- | --- | --- | --- | --- | --- | --- |
inator. W istrainedtopreventthediscriminatorfrommakingaccuratepredictions. Asaresult,this
isatwo-playergame,wherethediscriminatoraimsatmaximizingitsabilitytoidentifytheoriginof
anembedding,andW aimsatpreventingthediscriminatorfromdoingsobymakingWX andY as
similaraspossible. ThisapproachisinlinewiththeworkofGaninetal.(2016),whoproposedto
learnlatentrepresentationsinvarianttotheinputdomain,whereinourcase,adomainisrepresented
byalanguage(sourceortarget).
Discriminatorobjective Werefertothediscriminatorparametersasθ . Weconsidertheprob-
| (cid:0) | (cid:12) (cid:1) |     |     |     | D   |     |
| ------- | ---------------- | --- | --- | --- | --- | --- |
abilityP source = 1(cid:12)z thatavectorz isthemappingofasourceembedding(asopposedtoa
θD
targetembedding)accordingtothediscriminator. Thediscriminatorlosscanbewrittenas:
|     | n          |         |                  | m          |         |                  |
| --- | ---------- | ------- | ---------------- | ---------- | ------- | ---------------- |
|     | 1 (cid:88) | (cid:0) | (cid:12) (cid:1) | 1 (cid:88) | (cid:0) | (cid:12) (cid:1) |
L (θ |W)=− logP source=1(cid:12)Wx − logP source=0(cid:12)y . (3)
| D D | n   | θD  | i   | m   | θD  | i   |
| --- | --- | --- | --- | --- | --- | --- |
|     | i=1 |     |     | i=1 |     |     |
Mapping objective In the unsupervised setting, W is now trained so that the discriminator is
unabletoaccuratelypredicttheembeddingorigins:
|     | 1 n      |         |                  | 1 m      |         |                  |
| --- | -------- | ------- | ---------------- | -------- | ------- | ---------------- |
|     | (cid:88) | (cid:0) | (cid:12) (cid:1) | (cid:88) | (cid:0) | (cid:12) (cid:1) |
L (W|θ )=− logP source=0(cid:12)Wx − logP source=1(cid:12)y . (4)
| W D | n   | θD  | i   | m   | θD  | i   |
| --- | --- | --- | --- | --- | --- | --- |
|     | i=1 |     |     | i=1 |     |     |
3

PublishedasaconferencepaperatICLR2018
Learning algorithm To train our model, we follow the standard training procedure of deep ad-
versarial networks of Goodfellow et al. (2014). For every input sample, the discriminator and the
mapping matrix W are trained successively with stochastic gradient updates to respectively mini-
mizeL andL . Thedetailsoftrainingaregiveninthenextsection.
D W
2.2 REFINEMENTPROCEDURE
The matrix W obtained with adversarial training gives good performance (see Table 1), but the
results are still not on par with the supervised approach. In fact, the adversarial approach tries to
alignallwordsirrespectiveoftheirfrequencies. However,rarewordshaveembeddingsthatareless
updatedandaremorelikelytoappearindifferentcontextsineachcorpus,whichmakesthemharder
toalign.Undertheassumptionthatthemappingislinear,itisthenbettertoinfertheglobalmapping
using only the most frequent words as anchors. Besides, the accuracy on the most frequent word
pairsishighafteradversarialtraining.
Torefineourmapping, webuildasyntheticparallelvocabularyusingtheW justlearnedwithad-
versarialtraining. Specifically,weconsiderthemostfrequentwordsandretainonlymutualnearest
neighborstoensureahigh-qualitydictionary. Subsequently,weapplytheProcrustessolutionin(2)
on this generated dictionary. Considering the improved solution generated with the Procrustes al-
gorithm, it is possible to generate a more accurate dictionary and apply this method iteratively,
similarly to Artetxe et al. (2017). However, given that the synthetic dictionary obtained using ad-
versarialtrainingisalreadystrong,weonlyobservesmallimprovementswhendoingmorethanone
iteration,i.e.,theimprovementsonthewordtranslationtaskareusuallybelow1%.
2.3 CROSS-DOMAINSIMILARITYLOCALSCALING(CSLS)
Inthissubsection,ourmotivationistoproducereliablematchingpairsbetweentwolanguages: we
wanttoimprovethecomparisonmetricsuchthatthenearestneighborofasourceword,inthetarget
language,ismorelikelytohaveasanearestneighborthisparticularsourceword.
Nearestneighborsarebynatureasymmetric:ybeingaK-NNofxdoesnotimplythatxisaK-NN
of y. In high-dimensional spaces (Radovanovic´ et al., 2010), this leads to a phenomenon that is
detrimentaltomatchingpairsbasedonanearestneighborrule:somevectors,dubbedhubs,arewith
high probability nearest neighbors of many other points, while others (anti-hubs) are not nearest
neighbors of any point. This problem has been observed in different areas, from matching image
featuresinvision(Jegouetal.,2010)totranslatingwordsintextunderstandingapplications(Dinu
etal.,2015). Varioussolutionshavebeenproposedtomitigatethisissue,somebeingreminiscentof
pre-processingalreadyexistinginspectralclusteringalgorithms(Zelnik-manor&Perona,2005).
However, most studies aiming at mitigating hubness consider a single feature distribution. In our
case,wehavetwodomains,oneforeachlanguage.ThisparticularcaseistakenintoaccountbyDinu
et al. (2015), who propose a pairing rule based on reverse ranks, and the inverted soft-max (ISF)
by Smith et al. (2017), which we evaluate in our experimental section. These methods are not
fullysatisfactorybecausethesimilarityupdatesaredifferentforthewordsofthesourceandtarget
languages. Additionally,ISFrequirestocross-validateaparameter,whoseestimationisnoisyinan
unsupervisedsettingwherewedonothaveadirectcross-validationcriterion.
Incontrast,weconsiderabi-partiteneighborhoodgraph,inwhicheachwordofagivendictionary
isconnectedtoitsK nearestneighborsintheotherlanguage. WedenotebyN (Wx )theneigh-
T s
borhood, onthisbi-partitegraph, associatedwithamappedsourcewordembeddingWx . AllK
s
elements of N (Wx ) are words from the target language. Similarly we denote by N (y ) the
T s S t
neighborhoodassociatedwithawordtofthetargetlanguage. Weconsiderthemeansimilarityofa
sourceembeddingx toitstargetneighborhoodas
s
1 (cid:88)
r (Wx )= cos(Wx ,y ), (5)
T s K s t
yt∈NT(Wxs)
wherecos(.,.)isthecosinesimilarity. Likewisewedenotebyr (y )themeansimilarityofatarget
S t
wordy toitsneighborhood. Thesequantitiesarecomputedforallsourceandtargetwordvectors
t
withtheefficientnearestneighborsimplementationbyJohnsonetal.(2017). Weusethemtodefine
asimilaritymeasureCSLS(.,.)betweenmappedsourcewordsandtargetwords,as
CSLS(Wx ,y )=2cos(Wx ,y )−r (Wx )−r (y ). (6)
s t s t T s S t
4

PublishedasaconferencepaperatICLR2018
Intuitively,thisupdateincreasesthesimilarityassociatedwithisolatedwordvectors. Converselyit
decreasestheonesofvectorslyingindenseareas.OurexperimentsshowthattheCSLSsignificantly
increasestheaccuracyforwordtranslationretrieval,whilenotrequiringanyparametertuning.
3 TRAINING AND ARCHITECTURAL CHOICES
3.1 ARCHITECTURE
WeuseunsupervisedwordvectorsthatweretrainedusingfastText2. Thesecorrespondtomonolin-
gualembeddingsofdimension300trainedonWikipediacorpora;therefore,themappingW hassize
300×300.Wordsarelower-cased,andthosethatappearlessthan5timesarediscardedfortraining.
Asapost-processingstep,weonlyselectthefirst200kmostfrequentwordsinourexperiments.
For our discriminator, we use a multilayer perceptron with two hidden layers of size 2048, and
Leaky-ReLU activation functions. The input to the discriminator is corrupted with dropout noise
witharateof0.1. AssuggestedbyGoodfellow(2016),weincludeasmoothingcoefficients=0.2
in the discriminator predictions. We use stochastic gradient descent with a batch size of 32, a
learningrateof0.1andadecayof0.95bothforthediscriminatorandW. Wedividethelearning
rateby2everytimeourunsupervisedvalidationcriteriondecreases.
3.2 DISCRIMINATORINPUTS
Theembeddingqualityofrarewordsisgenerallynotasgoodastheoneoffrequentwords(Luong
et al., 2013), and we observed that feeding the discriminator with rare words had a small, but not
negligiblenegativeimpact.Asaresult,weonlyfeedthediscriminatorwiththe50,000mostfrequent
words.Ateachtrainingstep,thewordembeddingsgiventothediscriminatoraresampleduniformly.
Samplingthemaccordingtothewordfrequencydidnothaveanynoticeableimpactontheresults.
3.3 ORTHOGONALITY
Smith et al. (2017) showed that imposing an orthogonal constraint to the linear operator led to
better performance. Using an orthogonal matrix has several advantages. First, it ensures that the
monolingual quality of the embeddings is preserved. Indeed, an orthogonal matrix preserves the
dot product of vectors, as well as their (cid:96) distances, and is therefore an isometry of the Euclidean
2
space(suchasarotation). Moreover,itmadethetrainingproceduremorestableinourexperiments.
Inthiswork,weproposetouseasimpleupdatesteptoensurethatthematrixW staysclosetoan
orthogonalmatrixduringtraining(Cisseetal.(2017)). Specifically,wealternatetheupdateofour
modelwiththefollowingupdateruleonthematrixW:
W ←(1+β)W −β(WWT)W (7)
whereβ =0.01isusuallyfoundtoperformwell. Thismethodensuresthatthematrixstayscloseto
themanifoldoforthogonalmatricesaftereachupdate. Inpractice,weobservethattheeigenvalues
ofourmatricesallhaveamoduluscloseto1,asexpected.
3.4 DICTIONARYGENERATION
Therefinementsteprequirestogenerateanewdictionaryateachiteration.InorderfortheProcrustes
solutiontoworkwell,itisbesttoapplyitoncorrectwordpairs.Asaresult,weusetheCSLSmethod
describedinSection2.3toselectmoreaccuratetranslationpairsinthedictionary. Toincreaseeven
morethe qualityof thedictionary, andensure that W islearned fromcorrect translationpairs, we
only consider mutual nearest neighbors, i.e. pairs of words that are mutually nearest neighbors of
eachotheraccordingtoCSLS.Thissignificantlydecreasesthesizeofthegenerateddictionary,but
improvesitsaccuracy,aswellastheoverallperformance.
3.5 VALIDATIONCRITERIONFORUNSUPERVISEDMODELSELECTION
Selectingthebestmodelisachallenging,yetimportanttaskintheunsupervisedsetting,asitisnot
possibletouseavalidationset(usingavalidationsetwouldmeanthatwepossessparalleldata). To
2Wordvectorsdownloadedfrom:https://github.com/facebookresearch/fastText
5

PublishedasaconferencepaperatICLR2018
Figure 2: Unsupervised model selection.
Correlation between our unsupervised vali-
dationcriterion(blackline)andactualword
translationaccuracy(blueline). Inthispar-
ticular experiment, the selected model is at
epoch10. Observehowourcriterioniswell
correlatedwithtranslationaccuracy.
address this issue, we perform model selection using an unsupervised criterion that quantifies the
closenessofthesourceandtargetembeddingspaces.Specifically,weconsiderthe10kmostfrequent
sourcewords,anduseCSLStogenerateatranslationforeachofthem.Wethencomputetheaverage
cosinesimilaritybetweenthesedeemedtranslations,andusethisaverageasavalidationmetric. We
foundthatthissimplecriterionisbettercorrelatedwiththeperformanceontheevaluationtasksthan
optimaltransportdistancessuchastheWassersteindistance(Rubneretal.(2000)). Figure2shows
the correlation between the evaluation score and this unsupervised criterion (without stabilization
by learning rate shrinkage). We use it as a stopping criterion during training, and also for hyper-
parameterselectioninallourexperiments.
4 EXPERIMENTS
Inthissection,weempiricallydemonstratetheeffectivenessofourunsupervisedapproachonsev-
eral benchmarks, and compare it with state-of-the-art supervised methods. We first present the
cross-lingualevaluationtasksthatweconsidertoevaluatethequalityofourcross-lingualwordem-
beddings. Then, we present our baseline model. Last, we compare our unsupervised approach to
ourbaseline andtopreviousmethods. In theappendix, weoffer acomplementaryanalysison the
alignmentofseveralsetsofEnglishembeddingstrainedwithdifferentmethodsandcorpora.
4.1 EVALUATIONTASKS
Word translation The task considers the problem of retrieving the translation of given source
words.Theproblemwithmostavailablebilingualdictionariesisthattheyaregeneratedusingonline
toolslikeGoogleTranslate,anddonottakeintoaccountthepolysemyofwords. Failingtocapture
wordpolysemyinthevocabularyleadstoawrongevaluationofthequalityofthewordembedding
space. Otherdictionariesaregeneratedusingphrasetablesofmachinetranslationsystems,butthey
areverynoisyortrainedonrelativelysmallparallelcorpora. Forthistask, wecreatehigh-quality
en-es es-en en-fr fr-en en-de de-en en-ru ru-en en-zh zh-en en-eo eo-en
Methodswithcross-lingualsupervisionandfastTextembeddings
Procrustes-NN 77.4 77.3 74.9 76.1 68.4 67.7 47.0 58.2 40.6 30.2 22.1 20.4
Procrustes-ISF 81.1 82.6 81.1 81.3 71.1 71.5 49.5 63.8 35.7 37.5 29.0 27.9
Procrustes-CSLS 81.4 82.9 81.1 82.4 73.5 72.4 51.7 63.7 42.7 36.7 29.3 25.3
Methodswithoutcross-lingualsupervisionandfastTextembeddings
Adv-NN 69.8 71.3 70.4 61.9 63.1 59.6 29.1 41.5 18.5 22.3 13.5 12.1
Adv-CSLS 75.7 79.7 77.8 71.2 70.1 66.4 37.2 48.1 23.4 28.3 18.6 16.6
Adv-Refine-NN 79.1 78.1 78.1 78.2 71.3 69.6 37.3 54.3 30.9 21.9 20.7 20.6
Adv-Refine-CSLS 81.7 83.3 82.3 82.1 74.0 72.2 44.0 59.1 32.5 31.4 28.2 25.6
Table 1: Word translation retrieval P@1 for our released vocabularies in various language pairs. We
consider1,500sourcetestqueries,and200ktargetwordsforeachlanguagepair. WeusefastTextembeddings
trainedonWikipedia. NN:nearestneighbors. ISF:invertedsoftmax. (’en’isEnglish,’fr’isFrench,’de’is
German,’ru’isRussian,’zh’isclassicalChineseand’eo’isEsperanto)
6

PublishedasaconferencepaperatICLR2018
|     | EnglishtoItalian | ItaliantoEnglish |          |                 |      |
| --- | ---------------- | ---------------- | -------- | --------------- | ---- |
|     | P@1 P@5 P@10     | P@1 P@5 P@10     |          |                 |      |
|     |                  |                  | Table 2: | English-Italian | word |
Methodswithcross-lingualsupervision(WaCky) translationaverageprecisions(@1,
Mikolovetal.(2013b)† 33.8 48.3 53.9 24.9 41.0 47.4 @5, @10) from 1.5k source word
Dinuetal.(2015)† 38.5 56.4 63.9 24.6 45.4 54.1 queriesusing200ktargetwords.Re-
| CCA†               |                |                | sults marked                  | with the symbol | † are |
| ------------------ | -------------- | -------------- | ----------------------------- | --------------- | ----- |
|                    | 36.1 52.7 58.1 | 31.0 49.9 57.0 |                               |                 |       |
|                    |                |                | from Smith                    | et al. (2017).  | Wiki  |
| Artetxeetal.(2017) | 39.7 54.7 60.5 | 33.8 52.4 59.1 |                               |                 |       |
| Smithetal.(2017)†  |                |                | meanstheembeddingsweretrained |                 |       |
|                    | 43.1 60.7 66.4 | 38.0 58.5 63.6 |                               |                 |       |
|                    |                |                | on Wikipedia                  | using fastText. | Note  |
| Procrustes-CSLS    | 44.9 61.8 66.6 | 38.5 57.2 63.0 |                               |                 |       |
thatthemethodusedbyArtetxeetal.
Methodswithoutcross-lingualsupervision(WaCky)
(2017)doesnotusethesamesuper-
| Adv-Refine-CSLS | 45.1 60.7 65.1 | 38.3 57.8 62.8 |     |     |     |
| --------------- | -------------- | -------------- | --- | --- | --- |
visionasothersupervisedmethods,
Methodswithcross-lingualsupervision(Wiki) astheyonlyusenumbersintheirini-
Procrustes-CSLS 63.7 78.6 81.1 56.3 76.2 80.6 tialparalleldictionary.
Methodswithoutcross-lingualsupervision(Wiki)
| Adv-Refine-CSLS | 66.2 80.4 83.4 | 58.7 76.5 80.9 |     |     |     |
| --------------- | -------------- | -------------- | --- | --- | --- |
dictionariesofupto100kpairsofwordsusinganinternaltranslationtooltoalleviatethisissue. We
makethesedictionariespubliclyavailableaspartoftheMUSElibrary3.
Wereportresultsonthesebilingualdictionaries,aswellonthosereleasedbyDinuetal.(2015)to
allowforadirectcomparisonwithpreviousapproaches. Foreachlanguagepair,weconsider1,500
querysourceand200ktargetwords. Followingstandardpractice,wemeasurehowmanytimesone
ofthecorrecttranslationsofasourcewordisretrieved,andreportprecision@kfork =1,5,10.
Cross-lingual semantic word similarity We also evaluate the quality of our cross-lingual word
embeddings space using word similarity tasks. This task aims at evaluating how well the cosine
similaritybetweentwowordsofdifferentlanguagescorrelateswithahuman-labeledscore. Weuse
theSemEval2017competitiondata(Camacho-Colladosetal.(2017))whichprovideslarge, high-
quality and well-balanced datasets composed of nominal pairs that are manually scored according
| toawell-definedsimilarityscale. | WereportPearsoncorrelation. |     |     |     |     |
| ------------------------------- | --------------------------- | --- | --- | --- | --- |
Sentence translation retrieval Going from the word to the sentence level, we consider bag-of-
wordsaggregationmethodstoperformsentenceretrievalontheEuroparlcorpus.Weconsider2,000
sourcesentencequeriesand200ktargetsentencesforeachlanguagepairandreporttheprecision@k
for k = 1,5,10, which accounts for the fraction of pairs for which the correct translation of the
sourcewordsisinthek-thnearestneighbors. Weusetheidf-weightedaveragetomergewordinto
sentenceembeddings. Theidfweightsareobtainedusingother300ksentencesfromEuroparl.
4.2 RESULTSANDDISCUSSION
In what follows, we present the results on word translation retrieval using our bilingual dictionar-
ies in Table 1 and our comparison to previous work in Table 2 where we significantly outperform
previous approaches. We also present results on the sentence translation retrieval task in Table 3
andthecross-lingualwordsimilaritytaskinTable4. Finally, wepresentresultsonword-by-word
translationforEnglish-EsperantoinTable5.
Baselines In our experiments, we consider a supervised baseline that uses the solution of the
Procrustesformulagivenin(2),andtrainedonadictionaryof5,000sourcewords.Thisbaselinecan
becombinedwithdifferentsimilaritymeasures:NNfornearestneighborsimilarity,ISFforInverted
SoftMaxandtheCSLSapproachdescribedinSection2.2.
Cross-domainsimilaritylocalscaling ThisapproachhasasingleparameterK definingthesize
oftheneighborhood.TheperformanceisverystableandthereforeKdoesnotneedcross-validation:
theresultsareessentiallythesameforK =5,10and50,thereforewesetK =10inallexperiments.
InTable1,weobservetheimpactofthesimilaritymetricwiththeProcrustessupervisedapproach.
Looking at the difference between Procrustes-NN and Procrustes-CSLS, one can see that CSLS
3https://github.com/facebookresearch/MUSE
7

PublishedasaconferencepaperatICLR2018
EnglishtoItalian ItaliantoEnglish
P@1 P@5 P@10 P@1 P@5 P@10
Table 3: English-Italian sentence
Methodswithcross-lingualsupervision translation retrieval. We report
Mikolovetal.(2013b)† 10.5 18.7 22.8 12.0 22.1 26.7 theaverageP@kfrom2,000source
Dinuetal.(2015)† 45.3 72.4 80.7 48.9 71.3 78.3 queries using 200,000 target sen-
Smithetal.(2017)† 54.6 72.7 78.2 42.9 62.2 69.2 tences.Weusethesameembeddings
Procrustes-NN 42.6 54.7 59.0 53.5 65.5 69.5 as in Smith et al. (2017). Their re-
Procrustes-CSLS 66.1 77.1 80.7 69.5 79.6 83.5 sultsaremarkedwiththesymbol†.
Methodswithoutcross-lingualsupervision
Adv-CSLS 42.5 57.6 63.6 47.0 62.1 67.8
Adv-Refine-CSLS 65.9 79.7 83.1 69.0 79.7 83.1
providesastrongandrobustgaininperformanceacrossalllanguagepairs,withupto7.2%inen-
eo. We observe that Procrustes-CSLS is almost systematically better than Procrustes-ISF, while
being computationally faster and not requiring hyper-parameter tuning. In Table 2, we compare
ourProcrustes-CSLSapproachtopreviousmodelspresentedinMikolovetal.(2013b);Dinuetal.
(2015); Smith et al. (2017); Artetxe et al. (2017) on the English-Italian word translation task, on
which state-of-the-art models have been already compared. We show that our Procrustes-CSLS
approach obtains an accuracy of 44.9%, outperforming all previous approaches. In Table 3, we
alsoobtainastronggaininaccuracyintheItalian-EnglishsentenceretrievaltaskusingCSLS,from
53.5%to69.5%,outperformingpreviousapproachesbyanabsolutegainofmorethan20%.
Impactofthemonolingualembeddings Forthewordtranslationtask,weobtainedasignificant
boost in performance when considering fastText embeddings trained on Wikipedia, as opposed to
previously used CBOW embeddings trained on the WaCky datasets (Baroni et al. (2009)), as can
beenseeninTable2. Amongthetwofactorsofvariation,wenoticedthatthisboostinperformance
wasmostlyduetothechangeincorpora. ThefastTextembeddings,whichincorporatesmoresyn-
tacticinformationaboutthewords,obtainedonlytwopercentmoreaccuracycomparedtoCBOW
embeddingstrainedonthesamecorpus,outofthe18.8%gain. Wehypothesizethatthisgainisdue
tothesimilarco-occurrencestatisticsofWikipediacorpora. Figure3intheappendixshowsresults
on the alignment of different monolingual embeddings and concurs with this hypothesis. We also
obtainedbetterresultsformonolingualevaluationtaskssuchaswordsimilaritiesandwordanalogies
whentrainingourembeddingsontheWikipediacorpora.
Adversarialapproach Table1showsthattheadversarialapproachprovidesastrongsystemfor
learning cross-lingual embeddings without parallel data. On the es-en and en-fr language pairs,
Adv-CSLS obtains a P@1 of 79.7% and 77.8%, which is only 3.2% and 3.3% below the super-
vised approach. Additionally, we observe that most systems still obtain decent results on distant
languages that do not share a common alphabet (en-ru and en-zh), for which method exploiting
identicalcharacterstringsarejustnotapplicable(Artetxeetal.(2017)). Thismethodallowsusto
buildastrongsyntheticvocabularyusingsimilaritiesobtainedwithCSLS.Thegaininabsoluteac-
curacyobservedwithCSLSontheProcrustesmethodisevenmoreimportanthere,withdifferences
between Adv-NN and Adv-CSLS of up to 8.4% on es-en. As a simple baseline, we tried to match
the first two moments of the projected source and target embeddings, which amounts to solving
W(cid:63) ∈ argmin (cid:107)(WX)T(WX) − YTY(cid:107) and solving the sign ambiguity (Umeyama, 1988).
W F
This attempt was not successful, which we explain by the fact that this method tries to align only
thefirsttwomoments,whileadversarialtrainingmatchesallthemomentsandcanlearntofocuson
specificareasofthedistributionsinsteadofconsideringglobalstatistics.
Refinement: closingthegapwithsupervisedapproaches Therefinementsteponthesynthetic
bilingualvocabularyconstructedafteradversarialtrainingbringsanadditionalandsignificantgain
inperformance,closingthegapbetweenourapproachandthesupervisedbaseline. InTable1,we
observethatourunsupervisedmethodevenoutperformsourstrongsupervisedbaselineonen-itand
en-es,andisabletoretrievethecorrecttranslationofasourcewordwithupto83%accuracy. The
better performance of the unsupervised approach can be explained by the strong similarity of co-
occurrence statistics between the languages, and by the limitation in the supervised approach that
usesapre-definedfixed-sizevocabulary(of5,000uniquesourcewords): inourcasetherefinement
stepcanpotentiallyusemoreanchorpoints. InTable3,wealsoobserveastronggaininaccuracy
8

PublishedasaconferencepaperatICLR2018
SemEval2017 en-es en-de en-it
Methodswithcross-lingualsupervision en-eo eo-en
NASARI 0.64 0.60 0.65 Dictionary-NN 6.1 11.9
ourbaseline 0.72 0.72 0.71 Dictionary-CSLS 11.1 14.3
Methodswithoutcross-lingualsupervision Table5:BLEUscoreonEnglish-Esperanto.
Adv 0.69 0.70 0.67 Although being a naive approach, word-by-
Adv-Refine 0.71 0.71 0.71 wordtranslationisenoughtogetaroughidea
Table 4: Cross-lingual wordsim task. NASARI oftheinputsentence.Thequalityofthegener-
(Camacho-Colladosetal.(2016))referstotheofficial ateddictionaryhasasignificantimpactonthe
SemEval2017baseline.WereportPearsoncorrelation. BLEUscore.
(upto15%)onsentenceretrievalusingbag-of-wordsembeddings,whichisconsistentwiththegain
observedonthewordretrievaltask.
Application to a low-resource language pair and to machine translation Our method is par-
ticularly suited for low-resource languages for which there only exists a very limited amount of
paralleldata. WeapplyittotheEnglish-Esperantolanguagepair. WeusethefastTextembeddings
trainedonWikipedia, andcreateadictionarybasedonanonlinelexicon. Theperformanceofour
unsupervisedapproachonEnglish-Esperantoisof28.2%, comparedto29.3%withthesupervised
method. On Esperanto-English, our unsupervised approach obtains 25.6%, which is 1.3% better
thanthesupervisedmethod. Thedictionaryweuseforthatlanguagepairdoesnottakeintoaccount
thepolysemyofwords,whichexplainswhytheresultsarelowerthanonotherlanguagepairs. Peo-
plecommonlyreporttheP@5toalleviatethisissue. Inparticular,theP@5forEnglish-Esperanto
andEsperanto-Englishisof46.5%and43.9%respectively.
Toshowtheimpactofsuchadictionaryonmachinetranslation,weapplyittotheEnglish-Esperanto
Tatoebacorpora(Tiedemann,2012).Weremoveallpairscontainingsentenceswithunknownwords,
resulting in about 60k pairs. Then, we translate sentences in both directions by doing word-by-
wordtranslation. InTable5,wereporttheBLEUscorewiththismethod,whenusingadictionary
generatedusingnearestneighbors,andCSLS.WithCSLS,thisnaiveapproachobtains11.1and14.3
BLEU on English-Esperanto and Esperanto-English respectively. Table 6 in the appendix shows
some examples of sentences in Esperanto translated into English using word-by-word translation.
As one can see, the meaning is mostly conveyed in the translated sentences, but the translations
containsomesimpleerrors. Forinstance,the“mi”istranslatedinto“sorry”insteadof“i”,etc. The
translationscouldeasilybeimprovedusingalanguagemodel.
5 RELATED WORK
Workonbilinguallexiconinductionwithoutparallelcorporahasalongtradition,startingwiththe
seminal works by Rapp (1995) and Fung (1995). Similar to our approach, they exploit the Harris
(1954)distributionalstructure,butusingdiscretewordrepresentationssuchasTF-IDFvectors. Fol-
lowingstudiesbyFung&Yee(1998);Rapp(1999);Schafer&Yarowsky(2002);Koehn&Knight
(2002); Haghighi et al. (2008); Irvine & Callison-Burch (2013) leverage statistical similarities be-
tweentwolanguagestolearnsmalldictionariesofafewhundredwords. Thesemethodsneedtobe
initializedwithaseedbilinguallexicon,usingforinstancetheeditdistancebetweensourceandtar-
getwords. Thiscanbeseenaspriorknowledge,onlyavailableforcloselyrelatedlanguages. There
isalsoalargeamountofstudiesinstatisticaldecipherment,wherethemachinetranslationproblem
is reduced to a deciphering problem, and the source language is considered as a ciphertext (Ravi
&Knight,2011;Pourdamghani&Knight,2017). Althoughinitiallynotbasedondistributionalse-
mantics,recentstudiesshowthattheuseofwordembeddingscanbringsignificantimprovementin
statisticaldecipherment(Douetal.,2015).
Theriseofdistributedwordembeddingshasrevivedsomeoftheseapproaches,nowwiththegoal
ofaligningembeddingspacesinsteadofjustaligningvocabularies. Cross-lingualwordembeddings
can be used to extract bilingual lexicons by computing the nearest neighbor of a source word, but
alsoallowotherapplicationssuchassentenceretrievalorcross-lingualdocumentclassification(Kle-
mentievetal.,2012). Ingeneral,theyareusedasbuildingblocksforvariouscross-linguallanguage
processingsystems. Morerecently,severalapproacheshavebeenproposedtolearnbilingualdictio-
nariesmappingfromthesourcetothetargetspace(Mikolovetal.,2013b;Zouetal.,2013;Faruqui
9

PublishedasaconferencepaperatICLR2018
& Dyer, 2014; Ammar et al., 2016). In particular, Xing et al. (2015) showed that adding an or-
thogonalityconstrainttothemappingcansignificantlyimproveperformance,andhasaclosed-form
solution. ThisapproachwasfurtherreferredtoastheProcrustesapproachinSmithetal.(2017).
The hubness problem for cross-lingual word embedding spaces was investigated by Dinu et al.
(2015). The authors added a correction to the word retrieval algorithm by incorporating a nearest
neighbors reciprocity term. More similar to our cross-domain similarity local scaling approach,
Smith et al. (2017) introduced the inverted-softmax to down-weight similarities involving often-
retrievedhubwords. Intuitively,givenaquerysourcewordandacandidatetargetword,theyesti-
matetheprobabilitythatthecandidatetranslatesbacktothequery,ratherthantheprobabilitythat
thequerytranslatestothecandidate.
Recent work by Smith et al. (2017) leveraged identical character strings in both source and target
languages to create a dictionary with low supervision, on which they applied the Procrustes al-
gorithm. Similar to this approach, recent work by Artetxe et al. (2017) used identical digits and
numberstoformaninitialseeddictionary,andperformedanupdatesimilartoourrefinementstep,
butiterativelyuntilconvergence. Whiletheyshowedtheycouldobtaingoodresultsusingaslittle
as twenty parallel words, their method still needs cross-lingual information and is not suitable for
languagesthatdonotshareacommonalphabet. Forinstance, themethodofArtetxeetal.(2017)
onourdatasetdoesnotworkonthewordtranslationtaskforanyofthelanguagepairs,becausethe
digitswerefilteredoutfromthedatasetsusedtotrainthefastTextembeddings. ThisiterativeEM-
based algorithm initialized with a seed lexicon has also been explored in other studies (Haghighi
etal.,2008;Kondraketal.,2017).
Therehasbeenafewattemptstoalignmonolingualwordvectorspaceswithnosupervision.Similar
toourwork,Zhangetal.(2017b)employedadversarialtraining,buttheirapproachisdifferentthan
ours in multiple ways. First, they rely on sharp drops of the discriminator accuracy for model
selection. In our experiments, their model selection criterion does not correlate with the overall
model performance, as shown in Figure 2. Furthermore, it does not allow for hyper-parameters
tuning,sinceitselectsthebestmodeloverasingleexperiment. Weargueitisaseriouslimitation,
sincethebesthyper-parametersvarysignificantlyacrosslanguagepairs. Despiteconsideringsmall
vocabulariesofafewthousandwords,theirmethodobtainedweakresultscomparedtosupervised
approaches. More recently, Zhang et al. (2017a) proposed to minimize the earth-mover distance
afteradversarialtraining. Theycomparetheirresultsonlytotheirsupervisedbaselinetrainedwith
asmallseedlexicon,whichisonetotwoordersofmagnitudesmallerthanwhatwereporthere.
6 CONCLUSION
In this work, we show for the first time that one can align word embedding spaces without any
cross-lingualsupervision,i.e.,solelybasedonunaligneddatasetsofeachlanguage,whilereaching
oroutperformingthequalityofprevioussupervisedapproachesinseveralcases. Usingadversarial
training, we are able to initialize a linear mapping between a source and a target space, which we
alsousetoproduceasyntheticparalleldictionary. Itisthenpossibletoapplythesametechniques
proposedforsupervisedtechniques,namelyaProcrusteanoptimization. Twokeyingredientscon-
tributetothesuccessofourapproach:Firstweproposeasimplecriterionthatisusedasaneffective
unsupervisedvalidationmetric. SecondweproposethesimilaritymeasureCSLS,whichmitigates
the hubness problem and drastically increases the word translation accuracy. As a result, our ap-
proachproduceshigh-qualitydictionariesbetweendifferentpairsoflanguages,withupto83.3%on
theSpanish-Englishwordtranslationtask. Thisperformanceisonparwithsupervisedapproaches.
OurmethodisalsoeffectiveontheEnglish-Esperantopair,therebyshowingthatitworksforlow-
resourcelanguagepairs,andcanbeusedasafirststeptowardsunsupervisedmachinetranslation.
ACKNOWLEDGMENTS
We thank Juan Miguel Pino, Moustapha Cisse´, Nicolas Usunier, Yann Ollivier, David Lopez-Paz,
AlexandreSablayrolles,andtheFAIRteamforusefulcommentsanddiscussions.
REFERENCES
Waleed Ammar, George Mulcaire, Yulia Tsvetkov, Guillaume Lample, Chris Dyer, and Noah A
Smith. Massivelymultilingualwordembeddings. arXivpreprintarXiv:1602.01925,2016.
10

PublishedasaconferencepaperatICLR2018
MikelArtetxe,GorkaLabaka,andEnekoAgirre. Learningprincipledbilingualmappingsofword
embeddingswhilepreservingmonolingualinvariance. ProceedingsofEMNLP,2016.
Mikel Artetxe, Gorka Labaka, and Eneko Agirre. Learning bilingual word embeddings with (al-
most) no bilingual data. In Proceedings of the 55th Annual Meeting of the Association for
Computational Linguistics (Volume 1: Long Papers), pp. 451–462. Association for Computa-
tionalLinguistics,2017.
MarcoBaroni, SilviaBernardini, AdrianoFerraresi, andErosZanchetta. Thewackywideweb: a
collection of very large linguistically processed web-crawled corpora. Language resources and
evaluation,43(3):209–226,2009.
Piotr Bojanowski, Edouard Grave, Armand Joulin, and Tomas Mikolov. Enriching word vectors
with subword information. Transactions of the Association for Computational Linguistics, 5:
135–146,2017.
Jose´ Camacho-Collados,MohammadTaherPilehvar,andRobertoNavigli. Nasari: Integratingex-
plicit knowledge and corpus statistics for a multilingual representation of concepts and entities.
ArtificialIntelligence,240:36–64,2016.
JoseCamacho-Collados,MohammadTaherPilehvar,NigelCollier,andRobertoNavigli. Semeval-
2017 task 2: Multilingual and cross-lingual semantic word similarity. Proceedings of the 11th
InternationalWorkshoponSemanticEvaluation(SemEval2017),2017.
HailongCao,TiejunZhao,ShuZhang,andYaoMeng.Adistribution-basedmodeltolearnbilingual
wordembeddings. ProceedingsofCOLING,2016.
MoustaphaCisse,PiotrBojanowski,EdouardGrave,YannDauphin,andNicolasUsunier. Parseval
networks: Improvingrobustnesstoadversarialexamples. InternationalConferenceonMachine
Learning,pp.854–863,2017.
GeorgianaDinu,AngelikiLazaridou,andMarcoBaroni.Improvingzero-shotlearningbymitigating
the hubness problem. International Conference on Learning Representations, Workshop Track,
2015.
QingDou,AshishVaswani,KevinKnight,andChrisDyer. Unifyingbayesianinferenceandvector
spacemodelsforimproveddecipherment. 2015.
LongDuong,HiroshiKanayama,TengfeiMa,StevenBird,andTrevorCohn. Learningcrosslingual
wordembeddingswithoutbilingualcorpora. ProceedingsofEMNLP,2016.
Manaal Faruqui and Chris Dyer. Improving vector space word representations using multilingual
correlation. ProceedingsofEACL,2014.
Pascale Fung. Compiling bilingual lexicon entries from a non-parallel english-chinese corpus. In
ProceedingsoftheThirdWorkshoponVeryLargeCorpora,pp.173–183,1995.
PascaleFungandLoYuenYee. Anirapproachfortranslatingnewwordsfromnonparallel,compa-
rabletexts. InProceedingsofthe17thInternationalConferenceonComputationalLinguistics-
Volume1,COLING’98,pp.414–420.AssociationforComputationalLinguistics,1998.
Yaroslav Ganin, Evgeniya Ustinova, Hana Ajakan, Pascal Germain, Hugo Larochelle, Franc¸ois
Laviolette, Mario Marchand, and Victor Lempitsky. Domain-adversarial training of neural net-
works. JournalofMachineLearningResearch,17(59):1–35,2016.
Ian Goodfellow. Nips 2016 tutorial: Generative adversarial networks. arXiv preprint
arXiv:1701.00160,2016.
Ian Goodfellow, Jean Pouget-Abadie, Mehdi Mirza, Bing Xu, David Warde-Farley, Sherjil Ozair,
AaronCourville,andYoshuaBengio.Generativeadversarialnets.Advancesinneuralinformation
processingsystems,pp.2672–2680,2014.
StephanGouws,YoshuaBengio,andGregCorrado. Bilbowa: Fastbilingualdistributedrepresenta-
tionswithoutwordalignments. InProceedingsofthe32ndInternationalConferenceonMachine
Learning(ICML-15),pp.748–756,2015.
11

PublishedasaconferencepaperatICLR2018
AriaHaghighi,PercyLiang,TaylorBerg-Kirkpatrick,andDanKlein. Learningbilinguallexicons
from monolingual corpora. In Proceedings of the 46th Annual Meeting of the Association for
ComputationalLinguistics,2008.
ZelligSHarris. Distributionalstructure. Word,10(2-3):146–162,1954.
AnnIrvineandChrisCallison-Burch. Supervisedbilinguallexiconinductionwithmultiplemono-
lingualsignals. InHLT-NAACL,2013.
Herve Jegou, Cordelia Schmid, Hedi Harzallah, and Jakob Verbeek. Accurate image search us-
ing the contextual dissimilarity measure. IEEE Transactions on Pattern Analysis and Machine
Intelligence,32(1):2–11,2010.
Jeff Johnson, Matthijs Douze, and Herve´ Je´gou. Billion-scale similarity search with gpus. arXiv
preprintarXiv:1702.08734,2017.
AlexandreKlementiev,IvanTitov,andBinodBhattarai. Inducingcrosslingualdistributedrepresen-
tationsofwords. ProceedingsofCOLING,pp.1459–1474,2012.
Philipp Koehn and Kevin Knight. Learning a translation lexicon from monolingual corpora. In
Proceedings of the ACL-02 workshop on Unsupervised lexical acquisition-Volume 9, pp. 9–16.
AssociationforComputationalLinguistics,2002.
GrzegorzKondrak,BradleyHauer,andGarrettNicolai. Bootstrappingunsupervisedbilinguallexi-
coninduction. InEACL,2017.
AngelikiLazaridou,GeorgianaDinu,andMarcoBaroni.Hubnessandpollution:Delvingintocross-
spacemappingforzero-shotlearning.Proceedingsofthe53rdAnnualMeetingoftheAssociation
forComputationalLinguistics,2015.
OmerLevyandYoavGoldberg. Neuralwordembeddingasimplicitmatrixfactorization. Advances
inneuralinformationprocessingsystems,pp.2177–2185,2014.
Thang Luong, Richard Socher, and Christopher D Manning. Better word representations with re-
cursiveneuralnetworksformorphology. CoNLL,pp.104–113,2013.
TomasMikolov,KaiChen,GregCorrado,andJeffreyDean. Efficientestimationofwordrepresen-
tationsinvectorspace. ProceedingsofWorkshopatICLR,2013a.
TomasMikolov, QuocVLe, andIlyaSutskever. Exploitingsimilaritiesamonglanguagesforma-
chinetranslation. arXivpreprintarXiv:1309.4168,2013b.
TomasMikolov,IlyaSutskever,KaiChen,GregSCorrado,andJeffDean. Distributedrepresenta-
tionsofwordsandphrasesandtheircompositionality.Advancesinneuralinformationprocessing
systems,pp.3111–3119,2013c.
Robert Parker, David Graff, Junbo Kong, Ke Chen, and Kazuaki Maeda. English gigaword.
LinguisticDataConsortium,2011.
JeffreyPennington,RichardSocher,andChristopherDManning. Glove: Globalvectorsforword
representation. ProceedingsofEMNLP,14:1532–1543,2014.
N.PourdamghaniandK.Knight. Decipheringrelatedlanguages. InEMNLP,2017.
MilosˇRadovanovic´,AlexandrosNanopoulos,andMirjanaIvanovic´.Hubsinspace:Popularnearest
neighborsinhigh-dimensionaldata. JournalofMachineLearningResearch,11(Sep):2487–2531,
2010.
Reinhard Rapp. Identifying word translations in non-parallel texts. In Proceedings of the 33rd
AnnualMeetingonAssociationforComputationalLinguistics,ACL’95,pp.320–322.Associa-
tionforComputationalLinguistics,1995.
Reinhard Rapp. Automatic identification of word translations from unrelated english and ger-
mancorpora. InProceedingsofthe37thAnnualMeetingoftheAssociationforComputational
Linguistics,ACL’99.AssociationforComputationalLinguistics,1999.
12

PublishedasaconferencepaperatICLR2018
S.RaviandK.Knight. Decipheringforeignlanguage. InACL,2011.
Yossi Rubner, Carlo Tomasi, and Leonidas J Guibas. The earth mover’s distance as a metric for
imageretrieval. Internationaljournalofcomputervision,40(2):99–121,2000.
CharlesSchaferandDavidYarowsky. Inducingtranslationlexiconsviadiversesimilaritymeasures
and bridge languages. In Proceedings of the 6th Conference on Natural Language Learning -
Volume20,COLING-02.AssociationforComputationalLinguistics,2002.
PeterHScho¨nemann. Ageneralizedsolutionoftheorthogonalprocrustesproblem. Psychometrika,
31(1):1–10,1966.
Samuel L Smith, David HP Turban, Steven Hamblin, and Nils Y Hammerla. Offline bilingual
wordvectors,orthogonaltransformationsandtheinvertedsoftmax. InternationalConferenceon
LearningRepresentations,2017.
Jrg Tiedemann. Parallel data, tools and interfaces in opus. In Nicoletta Calzolari (Conference
Chair),KhalidChoukri,ThierryDeclerck,MehmetUurDoan,BenteMaegaard,JosephMariani,
AsuncionMoreno,JanOdijk,andSteliosPiperidis(eds.),ProceedingsoftheEightInternational
Conference on Language Resources and Evaluation (LREC’12), Istanbul, Turkey, may 2012.
EuropeanLanguageResourcesAssociation(ELRA). ISBN978-2-9517408-7-7.
Shinji Umeyama. An eigendecomposition approach to weighted graph matching problems. IEEE
transactionsonpatternanalysisandmachineintelligence,10(5):695–703,1988.
Ivan Vulic and Marie-Francine Moens. Bilingual word embeddings from non-parallel document-
aligneddataappliedtobilinguallexiconinduction. Proceedingsofthe53rdAnnualMeetingof
theAssociationforComputationalLinguistics(ACL2015),pp.719–725,2015.
Chao Xing, Dong Wang, Chao Liu, and Yiye Lin. Normalized word embedding and orthogonal
transformforbilingualwordtranslation. ProceedingsofNAACL,2015.
LihiZelnik-manorandPietroPerona. Self-tuningspectralclustering. InL.K.Saul,Y.Weiss,and
L. Bottou (eds.), Advances in Neural Information Processing Systems 17, pp. 1601–1608. MIT
Press,2005.
MengZhang,YangLiu,HuanboLuan,andMaosongSun. Earthmover’sdistanceminimizationfor
unsupervised bilingual lexicon induction. In Proceedings of the 2017 Conference on Empirical
Methods in Natural Language Processing, pp. 1924–1935. Association for Computational Lin-
guistics,2017a.
Meng Zhang, Yang Liu, Huanbo Luan, and Maosong Sun. Adversarial training for unsupervised
bilingual lexicon induction. Proceedings of the 53rd Annual Meeting of the Association for
ComputationalLinguistics,2017b.
Will Y Zou, Richard Socher, Daniel M Cer, and Christopher D Manning. Bilingual word embed-
dingsforphrase-basedmachinetranslation. ProceedingsofEMNLP,2013.
13

PublishedasaconferencepaperatICLR2018
7 APPENDIX
Inordertogainabetterunderstandingoftheimpactofusingsimilarcorporaorsimilarwordem-
beddingmethods,weinvestigatedmergingtwoEnglishmonolingualembeddingspacesusingeither
WikipediaortheGigawordcorpus(Parkeretal.(2011)),andeitherSkip-Gram,CBOWorfastText
methods(seeFigure3).
| 100     |          |                     |           | 100       |                |
| ------- | -------- | ------------------- | --------- | --------- | -------------- |
| 100 100 | 100 99.9 | 99.9 99.9 99.7 99.6 | 99.7 99.7 | 99.7 99.7 | 99.2 99.3 98.5 |
96.2 97.3 96.3
| ycarucca	laveirter	droW 90 |     |     |     | ycarucca	laveirter	droW 90 |     |
| -------------------------- | --- | --- | --- | -------------------------- | --- |
90.1
87.3
| 80  |     |     |     | 80  |     |
| --- | --- | --- | --- | --- | --- |
| 70  |     |     |     | 70  |     |
| 60  |     |     |     | 60  |     |
| 50  |     |     |     | 50  |     |
| 40  |     |     |     | 40  |     |
5k	–7k 10k	–12k 50k	–52k 100k	–102k150k	–152k 5k	–7k 10k	–12k 50k	–52k 100k	–102k150k	–152k
|     | NN  | CSLS |     |     | NN CSLS |
| --- | --- | ---- | --- | --- | ------- |
(a)skip-gram-seed1(Wiki)→skip-gram-seed2(Wiki) (b)skip-gram(Wiki)→CBOW(Wiki)
| 100                        |      |      |      | 100                        |      |
| -------------------------- | ---- | ---- | ---- | -------------------------- | ---- |
| ycarucca	laveirter	droW 90 |      |      |      | ycarucca	laveirter	droW 90 |      |
| 89.8                       |      |      |      | 87.8                       |      |
| 87.3                       | 85.3 | 85.7 | 86.7 | 84.8                       |      |
| 80                         | 83   | 82.4 |      | 80                         | 82.9 |
80.4
|     |     | 77.7 |     |     | 78.5 |
| --- | --- | ---- | --- | --- | ---- |
| 70  |     |      |     | 70  |      |
|     |     | 71   |     |     | 70.8 |
69.3 67.9 67.3
| 60  |     |     |     | 60  |     |
| --- | --- | --- | --- | --- | --- |
57.6
| 50  |     |     |     | 50  |     |
| --- | --- | --- | --- | --- | --- |
48
| 40  |     |     |     | 40  |     |
| --- | --- | --- | --- | --- | --- |
5k	–7k 10k	–12k 50k	–52k 100k	–102k150k	–152k 5k	–7k 10k	–12k 50k	–52k 100k	–102k150k	–152k
|                                  | NN  | CSLS |     |                                   | NN CSLS |
| -------------------------------- | --- | ---- | --- | --------------------------------- | ------- |
| (c)fastText(Wiki)→fastText(Giga) |     |      |     | (d)skip-gram(Wiki)→fastText(Giga) |         |
Figure3:EnglishtoEnglishwordalignmentaccuracy.Evolutionofwordtranslationretrievalaccuracywith
regardtowordfrequency,usingeitherWikipedia(Wiki)ortheGigawordcorpus(Giga),andeitherskip-gram,
continuousbag-of-words(CBOW)orfastTextembeddings.Themodelcanlearntoperfectlyalignembeddings
trainedonthesamecorpusbutwithdifferentseeds(a),aswellasembeddingslearnedusingdifferentmodels
(overall, when employing CSLS which is more accurate on rare words) (b). However, the model has more
troublealigningembeddingstrainedondifferentcorpora(WikipediaandGigaword)(c).Thiscanbeexplained
bythedifferenceinco-occurrencestatisticsofthetwocorpora,particularlyontherarerwords. Performance
canbefurtherdeterioratedbyusingbothdifferentmodelsanddifferenttypesofcorpus(d).
|     | Source     | mikelkfojeparolaskunmianajbarotralabarilo.          |     |     |     |
| --- | ---------- | --------------------------------------------------- | --- | --- | --- |
|     | Hypothesis | sorrysometimesspeakswithmyneighboracrossthebarrier. |     |     |     |
|     | Reference  | isometimestalktomyneighboracrossthefence.           |     |     |     |
|     | Source     | laviromalantaililudaslapianon.                      |     |     |     |
|     | Hypothesis | themanbehindtheyplaysthepiano.                      |     |     |     |
|     | Reference  | themanbehindthemisplayingthepiano.                  |     |     |     |
|     | Source     | bonvoleprotektuminkontratiujmalbonajviroj.          |     |     |     |
|     | Hypothesis | gratefullyprotectshiagainstthoseworstmen.           |     |     |     |
|     | Reference  | pleasedefendmefromsuchbadmen.                       |     |     |     |
Table 6: Esperanto-English. Examples of fully unsupervised word-by-word translations. The translations
reflectthemeaningofthesourcesentences,andcouldpotentiallybeimprovedusingasimplelanguagemodel.
14
