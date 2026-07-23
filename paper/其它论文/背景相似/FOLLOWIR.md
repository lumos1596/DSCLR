|     | FOLLOWIR:    |              |                         | Evaluating        |     | and Teaching |                      | Information |     | Retrieval     |     |     |     |
| --- | ------------ | ------------ | ----------------------- | ----------------- | --- | ------------ | -------------------- | ----------- | --- | ------------- | --- | --- | --- |
|     |              |              |                         | Models            | to  | Follow       | Instructions         |             |     |               |     |     |     |
|     |              | OrionWellerι |                         | BenjaminChangι    |     |              | SeanMacAvaneyλ       |             |     | KyleLoα       |     |     |     |
|     | ArmanCohanγα |              |                         | BenjaminVanDurmeι |     |              | DawnLawrieι          |             |     | LucaSoldainiα |     |     |     |
|     |              |              | ιJohnsHopkinsUniversity |                   |     |              | αAllenInstituteforAI |             |     |               |     |     |     |
|     |              |              | λUniversityofGlasgow    |                   |     |              | γYaleUniversity      |             |     |               |     |     |     |
oweller@cs.jhu.edu
|     |     | Abstract |     |     |     |     | IncontrasttothebroaderLMcommunity,infor- |     |     |     |     |     |     |
| --- | --- | -------- | --- | --- | --- | --- | ---------------------------------------- | --- | --- | --- | --- | --- | --- |
mationretrieval(IR)practitionersandresearchers
ModernLanguageModels(LMs)arecapable
haveyettofullyexploitinstruction-tunedmodels.
offollowinglongandcomplexinstructionsthat
enablealargeanddiversesetofuserrequests. Thanks to their ability to effectively estimate se-
WhileInformationRetrieval(IR)modelsuse mantic similarity between query and documents,
these LMs as the backbone of their architec- LMs have been adopted as the main backbone
| tures, | virtually | none | of them | allow | users | to  |     |     |     |     |     |     |     |
| ------ | --------- | ---- | ------- | ----- | ----- | --- | --- | --- | --- | --- | --- | --- | --- |
ofneuralretrievalarchitectures(Karpukhinetal.,
providedetailedinstructionsalongsidequeries,
|             |          |               |         |            |          |     | 2020;     | Khattab | and Zaharia,    |     | 2020; Reimers |          | and |
| ----------- | -------- | ------------- | ------- | ---------- | -------- | --- | --------- | ------- | --------------- | --- | ------------- | -------- | --- |
| thus        | limiting | their ability |         | to satisfy | complex  |     |           |         |                 |     |               |          |     |
|             |          |               |         |            |          |     | Gurevych, |         | 2019). However, |     | the vast      | majority | of  |
| information |          | needs.        | In this | work,      | we study |     |           |         |                 |     |               |          |     |
thesesystemsarefine-tunedtooperateexclusively
| the | use | of instructions | in  | IR systems. | We  |     |     |     |     |     |     |     |     |
| --- | --- | --------------- | --- | ----------- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
buildFOLLOWIR,arigorousinstructioneval- astextspanssimilarityestimators(KhattabandZa-
haria,2020;Izacardetal.,2021;NogueiraandCho,
uationbenchmarkforfollowingreal-worldin-
structions in IR. FOLLOWIR repurposes de- 2019;Pradeepetal.,2023;Maetal.,2023). Mov-
tailedinstructions—alsoknownasnarratives—
|     |     |     |     |     |     |     | ing past | these | ad-hoc | search | systems | to retrieve |     |
| --- | --- | --- | --- | --- | --- | --- | -------- | ----- | ------ | ------ | ------- | ----------- | --- |
developedforprofessionalassessorstoevalu-
|     |           |          |                |     |          |     | with instructions |     | would  | enable | support  | for     | com- |
| --- | --------- | -------- | -------------- | --- | -------- | --- | ----------------- | --- | ------ | ------ | -------- | ------- | ---- |
| ate | retrieval | systems. | In particular, |     | we build |     |                   |     |        |        |          |         |      |
|     |           |          |                |     |          |     | plex information  |     | needs. | For    | example, | imagine | a    |
ourbenchmarkfromthreecollectionscurated
researcherseekingtoidentifypapersthatmustcon-
forsharedtasksattheTextREtrievalConfer-
ence (TREC). Through this process, we can tainnumerousqualitiestoberelevant(fromagiven
|         |     |             |        |        |          |     | venue,usingagivenclassofmethods,etc.) |     |     |     |     |     | while |
| ------- | --- | ----------- | ------ | ------ | -------- | --- | ------------------------------------- | --- | --- | --- | --- | --- | ----- |
| measure |     | how well IR | models | follow | instruc- |     |                                       |     |     |     |     |     |       |
tions,throughanewpairwiseevaluationframe- also making sure to avoid conditions that would
work.Ourresultsindicatethatexistingretrieval makeitnot-relevant(usingnegativesentiment,us-
| models |     | fail to correctly | use | instructions, | us- |     |     |     |     |     |     |     |     |
| ------ | --- | ----------------- | --- | ------------- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
ingdatasetsfromcertaindomains,etc.).
ingthemforbasickeywordsandstrugglingto
Recentworkhasstartedtomovetowardssearch
| understandlong-forminformation. |      |                     |     |               | However, |     |                    |     |     |            |          |            |     |
| ------------------------------- | ---- | ------------------- | --- | ------------- | -------- | --- | ------------------ | --- | --- | ---------- | -------- | ---------- | --- |
|                                 |      |                     |     |               |          |     | with instructions, |     | but | this topic | is still | understud- |     |
| we                              | show | that it is possible |     | for IR models |          | to  |                    |     |     |            |          |            |     |
learntofollowcomplexinstructions: ournew iedwithonlyahandfulofpapers(Suetal.,2022;
|     |     |     |     |     |     |     | Asai | et al., | 2022; Muennighoff |     | et al., | 2024). | In  |
| --- | --- | --- | --- | --- | --- | --- | ---- | ------- | ----------------- | --- | ------- | ------ | --- |
FOLLOWIR-7Bmodelhassignificantimprove-
mentsafterfine-tuningonourtrainingset.1 particular, we find their use of instructions to be
|     |     |     |     |     |     |     | narrow: | instructionsaretypicallyshort(fewerthan |     |     |     |     |     |
| --- | --- | --- | --- | --- | --- | --- | ------- | --------------------------------------- | --- | --- | --- | --- | --- |
1 Introduction
10words)andrepetitive(onlyoneinstructionper
Modern language models (LMs) are extensively datasete.g.,Suetal.(2022);Asaietal.(2022);Li
tuned to be able to follow user instructions faith- andLi(2023);Xiaoetal.(2023)). Further, these
fully (Chung et al., 2022; Ouyang et al., 2022a; workslackevaluationdatasetsthatexplicitlymea-
Rafailov et al., 2023; Wang et al., 2023b; Ivison sure instruction following—instead focusing on
et al., 2023) and safely (Bai et al., 2022; Bianchi standardad-hocretrievalbenchmarks.
et al., 2024). Through these capabilities, LMs Toaddressthesegapsweintroduce FOLLOWIR,
are able to successfully tackle a broad range of which consists of (1) a benchmark that explicitly
tasks(Chiangetal.,2024;Liangetal.,2023;Yang measures the instruction following ability of re-
etal.,2023;Jimenezetal.,2024;Zengetal.,2023), trievalmodels,and(2)trainingdatathatincludes
evenwhennotexplicitlyfine-tunedforthem.
|        |     |                 |     |            |           |     | diverseandrealisticinstructions. |          |              |     | Ourkeyintuition |             |     |
| ------ | --- | --------------- | --- | ---------- | --------- | --- | -------------------------------- | -------- | ------------ | --- | --------------- | ----------- | --- |
|        |     |                 |     |            |           |     | is to                            | leverage | instructions |     | developed       | for profes- |     |
| 1Links | to  | the code, data, | and | models are | available | at  |                                  |          |              |     |                 |             |     |
https://github.com/orionw/FollowIR sionalannotatorsofIRsystemsinordertostudy
11926
Proceedingsofthe2025ConferenceoftheNationsoftheAmericasChapteroftheAssociationforComputationalLinguistics:HumanLanguageTechnologies
(Volume1:LongPapers),pages11926–11942
April29-May4,2025©2025AssociationforComputationalLinguistics

Figure1: Howdostandardretrievalqueriesdifferfrominstructions(ornarratives)? Instructionscontainmore
specificdetailsaboutwhatisrelevant,includelessdirectly-relevantbackgroundinfo,andoftenhavedirectives
aboutwhatdocumentsarenotrelevant,usingnegation. %’sarehowoftenfeaturesappearintheTRECinstructions.
the capabilities of instruction-following IR mod- samequery. Resultson FOLLOWIR indicatethat
els. These instructions are used by annotators to currentmodelsgenerallyfailtofollowinstructions
judgedocumentrelevanceforagivenquery. For- inretrievalunlesstheyare3B+parametersorhave
tunately, the IR field is rich with such data, as notbeentrainedforretrieval. Ouranalysisshows
theseinstructions—alsoknownasnarratives—are thatthesefailuresareduetotwophenomena: (1)
createdforallqueriesinanywell-constructedIR models are not used to long instructions, and (2)
dataset. Inparticular,weusenarrativesdeveloped modelsusetheinstructionstodokeywordsearch.
forsharedtasksattheTextREtrievalConference. Tofurtherprogressinbuildingretrievalmodels
These instructions are thorough and complex, in- thatcanunderstandinstructions,webuildatrain-
cluding minute details about what makes a docu- ingsetofreal-worldhuman-usedinstructionsand
ment relevant vs not-relevant. Thus if annotators fine-tuneamodelonthem(FOLLOWIR-7B).Our
canusetheseTRECinstructionstoannotatedoc- resultsshowmarkedimprovementonFOLLOWIR
umentrelevance,soshouldinstruction-following forbothstandardIRmetricsandforp-MRR,indi-
retrieval models (example query and instruction catingastartingpointforfutureprogress.
pairsareshowninFigures1and2). Insummary,wecontributethefollowing: (1)a
Weusethreedeeply-judgedTRECcollectionsas benchmarkforevaluatinginstructionfollowingin
thebasisofourevaluationset: TRECRobust2004 retrieval(FOLLOWIR)consistingofhumanannota-
(Voorhees,2005),TRECCommonCore2017(Al- tionsontopofthreealreadyhighly-judgedcorpora,
lanetal.,2017),andTRECNews2021(Soboroff (2) analysis of why current models fail to under-
et al., 2020). These collections have been thor- standinstructions,and(3)trainingdataforteaching
oughlyannotatedinordertoevaluaterecallinre- retrievalmodelstofollowinstructionsalongwitha
trieval,withhundredstothousandsofdocuments newopen-sourcedIRmodel,FOLLOWIR-7B,that
judged as relevant or not-relevant. We take the canhandlelonginstructionsinIR.
instructions given to the professional annotators
andalterthemslightly,manuallyre-annotatingthe
2 RelatedWork
relevantdocuments. Wethenhavepairedinstruc-
tions,whichcanbeusedtotesthowmodelsreactto TRECConferences TheUnitedStatesNational
changedinstructions: wemeasureifmodelsupdate InstituteofScienceandTechnology(NIST)created
theirrelevantdocstomatchthealteredinstructions. theTRECorganizationin1993. EachyearTREC
As there are no existing methods to compare sponsorsmanytracks,orsharedtasks,onagiven
pairwise queries in IR, we develop a new evalu- dataset. Thesetracksrangefromavarietyoftopics:
ation framework to do so, measuring rank-wise anywherefromstandardad-hocretrievalonnews
score changes (which we call p-MRR) of docu- (Soboroff et al., 2018; Soboroff, 2021) to more
mentsgivenapairofdifferentinstructionswiththe complexdomainssuchaslegalretrieval(Oardetal.,
11927

Query: Identify positive
|     |     |     |     | Doc 1: The first pictures of the  |     |     |     | Original |     | Altered |     |
| --- | --- | --- | --- | --------------------------------- | --- | --- | --- | -------- | --- | ------- | --- |
accomplishments of the Hubble
emerging universe from the US Cosmic
| telescope since it was launched.  |     |     |     | Explorer (Cobe) satellite with ideas |     |     |     |     |     |     |     |
| --------------------------------- | --- | --- | --- | ------------------------------------ | --- | --- | --- | --- | --- | --- | --- |
from the Hubble Space Telescope, have
|                                |     |     |     |                                         |     |     |     | Doc 1 |     | Doc 1 |     |
| ------------------------------ | --- | --- | --- | --------------------------------------- | --- | --- | --- | ----- | --- | ----- | --- |
| Instruction: Documents are     |     |     |     | inspired new cosmological theories .... |     |     |     |       |     |       |     |
|                                |     |     |     |                                         |     |     |     | ...   |     | ...   |     |
| relevant that show  the Hubble |     |     |     |                                         | ... |     |     |       |     |       |     |
telescope has produced new data,
better quality data than previously
available, ....,  are relevant if they
|     |     |     |     | Doc N: Photographs of a giant        |     |     |     | Doc N |     | Doc N |     |
| --- | --- | --- | --- | ------------------------------------ | --- | --- | --- | ----- | --- | ----- | --- |
relied solely on the Hubble.
storm on Saturn taken by the Hubble
| Documents limited to the problems |     |     |     | Space Telescope reveal that the storm |     |     |     |     |     |     |     |
| --------------------------------- | --- | --- | --- | ------------------------------------- | --- | --- | --- | --- | --- | --- | --- |
of the telescope are be irrelevant.
has grown so much since it was
Details of repairs to the telescope
|                               |     |     |     | discovered in September ... it is several |     |     |     | Does the model change with |     |     |     |
| ----------------------------- | --- | --- | --- | ----------------------------------------- | --- | --- | --- | -------------------------- | --- | --- | --- |
| without reference to positive |     |     |     | times larger than Earth                   |     |     |     |                            |     |     |     |
| achievements would not  ...   |     |     |     |                                           |     |     |     | the altered instruction?   |     |     |     |
Figure2: Avisualdepictionofthepairwiseevaluationframework: modelsareevaluatedonthequerywiththe
original instruction, and then on the query with the altered instruction. If the model correctly understands the
instructions, it will change which documents are relevant w.r.t. the alteration (right). Note that the real-world
instructions(left)giventoTRECannotatorsincludesfine-graineddetailsabouttherelevanceaswellasnegation.
2008), or retrieval-augmented generation/report- Instructions for Retrieval Using instructions
generation(Lawrieetal.,2024). in retrieval is a nascent area of exploration. Su
Aspartofthisprocess, NISTsponsorsannota- et al. (2022) and Asai et al. (2022) were two of
tionsforthesecollections. Typically,thisisdone theearliestworksthattrainedaretrievalmodelto
|            |          |         |        |      |           | use instructions |     | along with | the | query. However, |     |
| ---------- | -------- | ------- | ------ | ---- | --------- | ---------------- | --- | ---------- | --- | --------------- | --- |
| by pooling | a set of | results | (runs) | from | a variety |                  |     |            |     |                 |     |
of retrieval models and then annotating them in theseinstructionsaretypicallyveryshort,suchas
rankorderuntilfundingrunsout. Tohelpfacilitate “RetrieveaWikipediaparagraphthatanswersthis
annotation, track organizers provide a narrative question."Recentworkincorporatesinstructionsin
(or instruction) for each query that will be given smallermodels(Xiaoetal.,2023;Chenetal.,2023,
to the annotators—however, IR models are only 2024)aswellasotherswhichuseLlama(Touvron
ever given the query. As evaluating total recall etal.,2023a;Welleretal.,2023)orMistral(Jiang
|       |                    |     |       |          |        | et al., 2023) | as  | the backbone | of  | a larger retrieval |     |
| ----- | ------------------ | --- | ----- | -------- | ------ | ------------- | --- | ------------ | --- | ------------------ | --- |
| would | require annotating |     | every | document | in the |               |     |              |     |                    |     |
collection for every query (which is not feasible model that can use instructions: GritLM (Muen-
forcollectionswithmillionsofdocuments),recall nighoffetal.,2024)trainsMistraltodobothgen-
erroristestedusingpost-hocsamplingandannota- erationandembedding,whileWangetal.(2023a)
tion. Althoughnoteveryqueryanddocumentpair usesMistralforembeddingsonly.
can be evaluated, recall for queries is very high. Despitethisflurryofactivity,theseeffortsdonot
WebuildofftherigorousevaluationdoneatTREC haveanexplicitinstruction-relatedretrievalbench-
bystartingwithseveraloftheircollections.
|     |     |     |     |     |     | mark to | evaluate | on. Instead, |     | they evaluate | on  |
| --- | --- | --- | --- | --- | --- | ------- | -------- | ------------ | --- | ------------- | --- |
Instructions for LMs Instruction-following standardretrievalbenchmarksuitessuchasMTEB
(Muennighoffetal.,2022)andBEIR(Thakuretal.,
| LMs have | been popularized |     | by  | models | such as |     |     |     |     |     |     |
| -------- | ---------------- | --- | --- | ------ | ------- | --- | --- | --- | --- | --- | --- |
InstructGPT (Ouyang et al., 2022a), FLAN (Wei 2021) which do not contain instructions. Thus,
et al., 2022), and T0 (Sanh et al., 2022). They thesenewerinstructionretrievalmodelshand-write
|     |     |     |     |     |     | a few instructions, |     | where | typically | each | instruc- |
| --- | --- | --- | --- | --- | --- | ------------------- | --- | ----- | --------- | ---- | -------- |
havebecomealargeareaofinterestforthenatural
language processing community (Touvron et al., tionisappliedtoanentiredataset,irrespectiveof
2023a;Jiangetal.,2023;Groeneveldetal.,2024; the query. This makes these instructions generic:
focusedonlyonthetaskformat,formatofthe“doc-
| Black et | al., 2022). | There | has | been much | work |     |     |     |     |     |     |
| -------- | ----------- | ----- | --- | --------- | ---- | --- | --- | --- | --- | --- | --- |
ument"(paragraph,sentence,etc.),andthebroad
inevaluatingiftheycangeneralizetonewinstruc-
tions (Wang et al., 2022c; Ouyang et al., 2022b), domain. Note that because of this, no current in-
if we can train them to follow instructions with- structionscontainanyextrabackgroundinforma-
|                     |     |      |       |         |        | tion or negation |     | (Weller | et al., | 2024) which | are |
| ------------------- | --- | ---- | ----- | ------- | ------ | ---------------- | --- | ------- | ------- | ----------- | --- |
| out human-annotated |     | data | (Wang | et al., | 2022b; |                  |     |         |         |             |     |
Qin et al., 2023), and applying them to various commonly found in real-world instructions (see
domains (Zhao et al., 2021; Singhal et al., 2023; Figure1foranexampleofthesedifferences).
Shaghaghian et al., 2020). As the IR community Inworkconcurrenttoours,Ohetal.(2024)also
usesLMsintheirpipelines,weseektobroadenthe propose a dataset to evaluate instructions in re-
scopeofIRtoincludeinstructions,aligningitwith trievalmodels. TheirdatasetusestheMSMARCO
thebroaderNLPcommunity. collection(Nguyenetal.,2016),anddiffersinsev-
11928

eral crucial aspects: it only has one relevant doc- We annotate a subset of the original TREC
umentperquery(e.g.,sparselyjudged),isGPT-4
|     |     |     |     |     |     |     | queries | due to | cost | and overlap: |     | we sample | 50  |
| --- | --- | --- | --- | --- | --- | --- | ------- | ------ | ---- | ------------ | --- | --------- | --- |
generatedandvalidated,focusesonthebackground queriesfromTRECRobust2004thatdonotover-
oftheuser(“Iamaschoolteachinglookingfor..."), lapwithTRECCommonCore(asCommonCore
andevaluatesusingthelowestscoreoverN instruc- used 50 queries from Robust04 on a new collec-
tions for the same query (measuring robustness). tion),and30queriesfromTRECNews2021. Ta-
Incontrast,weusehighly-judgedcorporatoensure ble1showsdatasetstatisticsofjudgeddocuments
we can measure recall, use professionally gener- and the final benchmark size. Annotators were
atedinstructions,havehuman-validatedrelevance askedtochangetheinstructionssothatthenumber
judgements,proposeanewpairedevaluationpro- ofrelevantdocumentswascutroughlyinhalf,thus
tocol,andprovideatrainingdatasetandmodelfor includingasizeablenumberofchangedrelevance
teachinginstruction-following. judgements. We note that although the number
|     |     |     |     |     |     |     | of queries | seems | small | by  | NLP | standards, | 30-50 |
| --- | --- | --- | --- | --- | --- | --- | ---------- | ----- | ----- | --- | --- | ---------- | ----- |
3 Building FOLLOWIR queriesisbotheffective(Webberetal.,2008)and
standardinIRduetotheexpenseofcarefulanno-
| We derive | FOLLOWIR |     | from | three | TREC | collec- |     |     |     |     |     |     |     |
| --------- | -------- | --- | ---- | ----- | ---- | ------- | --- | --- | --- | --- | --- | --- | --- |
tationovermanydocumentsperquery(see§A).
tions: TRECNews2021(derivedfromtheWash-
Duetodifferencesinretrieverquality,ifweeval-
| ington | Post v4 | corpus;    | Soboroff |          | et al., | 2020),   |         |           |     |          |      |             |      |
| ------ | ------- | ---------- | -------- | -------- | ------- | -------- | ------- | --------- | --- | -------- | ---- | ----------- | ---- |
|        |         |            |          |          |         |          | uate by | searching |     | over the | full | collection, | each |
| TREC   | Robust  | 2004 (from | news     | articles |         | in Disks |         |           |     |          |      |             |      |
modelwillretrieveadifferentnumberofrelevant
| 4 and 5   | collections; | Voorhees, |       | 2005),  | and    | TREC    |                |                                   |       |             |        |              |             |
| --------- | ------------ | --------- | ----- | ------- | ------ | ------- | -------------- | --------------------------------- | ----- | ----------- | ------ | ------------ | ----------- |
|           |              |           |       |         |        |         | documents.     | However,becauseweevaluateinstruc- |       |             |        |              |             |
| Common    | Core         | 2017      | (from | the New | York   | Times   |                |                                   |       |             |        |              |             |
|           |              |           |       |         |        |         | tion following |                                   | based | on changing |        | the document |             |
| Annotated | corpus;      | Allan     | et    | al.,    | 2017). | Each of |                |                                   |       |             |        |              |             |
|           |              |           |       |         |        |         | relevance,     | models                            | that  | do          | poorly | in the       | initial re- |
thesewasprofessionallyassessedtoincludehun-
|     |     |     |     |     |     |     | trieval will | have | fewer | documents |     | which | change |
| --- | --- | --- | --- | --- | --- | --- | ------------ | ---- | ----- | --------- | --- | ----- | ------ |
dredsofannotationsperquery(seeTable1),with
|        |          |           |     |           |     |         | relevance | in the | instruction-following |     |     | evaluation. |     |
| ------ | -------- | --------- | --- | --------- | --- | ------- | --------- | ------ | --------------------- | --- | --- | ----------- | --- |
| 50-180 | relevant | documents |     | per query | on  | average |           |        |                       |     |     |             |     |
Torectifythis,weinsteadturntoarerankingtask
(andmanymorenot-relevantannotations).
whereweincludeallrelevantdocuments,andusea
EachoftheseTRECtracksincludesinstructions
pooloffivemodels3
toselectthetopnon-relevant
| for the          | professional | annotators                 |     | that | we  | now also |            |                                   |     |     |     |     |     |
| ---------------- | ------------ | -------------------------- | --- | ---- | --- | -------- | ---------- | --------------------------------- | --- | --- | --- | --- | --- |
|                  |              |                            |     |      |     |          | documents. | Tobeabletofreelydistributethedata |     |     |     |     |     |
| givetothemodels. |              | Althoughusingthesealonecan |     |      |     |          |            |                                   |     |     |     |     |     |
duetofair-uselaws,wechunkthedocumentsinto
| provide | some | indication | of  | how well | models | can |     |     |     |     |     |     |     |
| ------- | ---- | ---------- | --- | -------- | ------ | --- | --- | --- | --- | --- | --- | --- | --- |
400-wordpassageswith200-wordoverlapandse-
| follow | instructions, | it  | doesn’t | explicitly |     | test their |     |     |     |     |     |     |     |
| ------ | ------------- | --- | ------- | ---------- | --- | ---------- | --- | --- | --- | --- | --- | --- | --- |
lectthehighestscoringpassagesusingMaxP(Dai
| instructionfollowingability. |     |     |     | Tomorecarefullyiso- |     |     |             |        |     |              |     |                  |     |
| ---------------------------- | --- | --- | --- | ------------------- | --- | --- | ----------- | ------ | --- | ------------ | --- | ---------------- | --- |
|                              |     |     |     |                     |     |     | and Callan, | 2019). |     | This enables |     | us to distribute |     |
latethisinourbenchmark,wetestwhethermodels
|     |     |     |     |     |     |     | our data, | which | we  | do by | extending | the | MTEB |
| --- | --- | --- | --- | --- | --- | --- | --------- | ----- | --- | ----- | --------- | --- | ---- |
canrespondtosmallchangesintheinstruction.
evaluationframework(Muennighoffetal.,2022).
Toaccomplishthis,weasktwoexpertannotators
| tomodifytheTRECinstructions. |     |     |     |     | However,doing |     |     |     |     |     |     |     |     |
| ---------------------------- | --- | --- | --- | --- | ------------- | --- | --- | --- | --- | --- | --- | --- | --- |
this in a naive way would require re-annotating 3.1 EvaluationMetricsfor FOLLOWIR
| all the       | document       | judgements,  |      | a          | non-trivial  | task     |                       |     |              |                             |          |              |     |
| ------------- | -------------- | ------------ | ---- | ---------- | ------------ | -------- | --------------------- | --- | ------------ | --------------------------- | -------- | ------------ | --- |
|               |                |              |      | efforts.2  |              |          | Our benchmark         |     | provides     |                             | two ways | of measuring |     |
| requiring     | immense        | annotation   |      |            |              | Instead, |                       |     |              |                             |          |              |     |
|               |                |              |      |            |              |          | instructionfollowing: |     |              | (1)standardretrievalmetrics |          |              |     |
| we task       | the annotators |              | with | making     | instructions |          |                       |     |              |                             |          |              |     |
|               |                |              |      |            |              |          | when using            | the | instructions |                             | with     | the queries  | and |
| more specific |                | by including |      | additional | constraints  |          |                       |     |              |                             |          |              |     |
(2)pairwiseevaluationofinstructionfollowing.
| thatnarrowtherelevancedefinition. |     |     |     |     | Thesetrans- |     |     |     |     |     |     |     |     |
| --------------------------------- | --- | --- | --- | --- | ----------- | --- | --- | --- | --- | --- | --- | --- | --- |
For(1),weusetypicalIRevaluationmetricsbut
| formations | cause | some | previously |     | relevant | docu- |     |     |     |     |     |     |     |
| ---------- | ----- | ---- | ---------- | --- | -------- | ----- | --- | --- | --- | --- | --- | --- | --- |
mentstobecomenon-relevantwithoutintroducing use the instruction along with the query: these
anynewrelevantdocumentsfromthepool. There- metrics are mean average precision (MAP) for
|     |     |     |     |     |     |     | Core17/Robust04 |     | and | normalized |     | discounted | cu- |
| --- | --- | --- | --- | --- | --- | --- | --------------- | --- | --- | ---------- | --- | ---------- | --- |
fore,onlythosedocumentsthatweredeemedrel-
|          |              |     |      |           |      |       | mulative | gain | at 5 (nDCG@5) |     | for | News21.4 | For |
| -------- | ------------ | --- | ---- | --------- | ---- | ----- | -------- | ---- | ------------- | --- | --- | -------- | --- |
| evant by | the original |     | TREC | assessors | need | to be |          |      |               |     |     |          |     |
re-annotated. Thismakestheannotationtractable, (2) we use our novel pairwise evaluation metric
withonlydozensofdocumentstore-annotateper
queryinsteadofacollectionofthousands. 3WeuseBM25,BGE-base,E5-base-v2,TART-Contriever,
andINSTRUCTOR-xl.
2NIST’s budget is $1–2 million USD/year: trec.nist. 4We use nDCG@5 for News21 as that was the official
| gov/pubs/2010.economic.impact.pdf |     |     |     |     |     |     | metricusedinthatTRECtrack. |     |     |     |     |     |     |
| --------------------------------- | --- | --- | --- | --- | --- | --- | -------------------------- | --- | --- | --- | --- | --- | --- |
11929

Dataset #Q Q I Rel.D/Q #Q I Rel.D/Q
| | | | | |
TRECNews’21(Soboroffetal.,2020) 50 15.3 40.1 50.1 32 46.9 19.2
TRECCore’17(Allanetal.,2017) 50 16.6 44.0 180.0 20 53.5 32.7
TRECRobust’04(Voorhees,2005) 249 11.9 68.2 69.9 52 75.2 19.8
Table1: FOLLOWIRevaluationsetstatisticsbefore(left)andafter(right)annotation. Weuseasubsetofthequeries
inthreepopularTRECtracksforvarietyinqueriesanddocuments. Q isthewordlengthofthequeriesand I is
| | | |
thewordlengthoftheinstructions. Rel. D/Qindicatesthenumberofrelevantannotateddocumentsinthecollection,
excludingirrelevantannotations. Asdesigned,therearelessrelevantly-judgeddocumentsintheFOLLOWIRportion
(astheannotationschangetherelevanceofdocumentsonpurposeforevaluation).
that measures the delta in scores when following 4 EvaluatingInstructionFollowing
themodifiedinstructionsinsteadoftheoriginal.5
In this section we describe the models we eval-
Our new pairwise evaluation metric, p-MRR,
uate, their results on FOLLOWIR, and ablations
measuresrank-wisechangesbetweenqueries. In
performedtobetterunderstandcurrentmodels.
developing this metric we had the following
desiderata: itshouldcomparetheresultsoftheorig-
4.1 EvaluationSettings
inal instruction to those of the new instruction, it
WeevaluateawidevarietyofIRmodels(trained
shouldhaveastandardizedrangefromworstpossi-
with and without instructions), including neural
blechangeininstruction-followingscore(i.e., 1)
− models ranging from 100 million to 7 billion pa-
tobestpossibleinstruction-followingscore(i.e.,1)
rameters. We evaluate on the original TREC in-
withanoptionfornochangewhenusingdifferent
instructions(i.e.,0),andfinallyshouldtakeintoac-
structionsintheFOLLOWIRbenchmarkandthen
onthenewinstructions,showingbothstandardIR
countthedocumentranksothatchangesfromrank
metricsandthenewpairwisemetricp-MRR.We
1torank2aremoreprominentthanchangesfrom
groupmodelsintofourcategories:
rank99to100. Giventheabovequalifications,we
usethefollowingequationappliedtoeachchanged No Instructions in Training These retrieval
relevancedocumentperquery(whereRRisrecip- models did not see instructions in training and
rocalrank,R istherankofthedocwhenusing typically aren’t given them: Contriever (Izacard
og
theoriginalinstructionandR isthenewrank): etal.,2021),E5(Wangetal.,2022a),MonoBERT
new
(Nogueiraetal.,2019),MonoT5(Nogueiraetal.,
RRog 1 ifR > R 2020),andBM25(Robertsonetal.,1995).
RRnew − og new
p-MRR = (1) InstructionsinIRTraining Mostretrievalmod-

 1 − R R R R n o e g w otherwise elsusinginstructionsreceivedroughlyoneinstruc-
tionperretrievaldataset,whichgenerallydefined

Forthefinalscore,weaveragefirstwithinagiven thedomain(e.g.,“Financial"),documentsize(sen-
query and then over all queries in the corpora— tence, passage, etc.), and task format. This in-
i.e.,macro-averagingacrossqueries,tohandlethe cludesINSTRUCTORmodels(Suetal.,2022),the
differentnumberofrelevantdocumentsperquery. bi-encoder TART model trained from Contriever
(Asaietal.,2022),thererankerTARTtrainedfrom
3.2 Whydoweneedanewmetric? FLAN-T5(Chungetal.,2022),E5Mistral-Instruct
(Wang et al., 2023a), and GritLM (Muennighoff
The original TREC datasets do not need instruc-
etal.,2024). WealsoincludeBGEmodels(Xiao
tions to be correctly solved. The nDCG metric
et al., 2023) in this category, although they are
onlyshowstheirabilitytoretrieveontheoriginal
trained with only one instruction total for each
task(withtheaddedinstruction). However,models
broadtask(retrieval,clustering,etc.).
could simply ignore the instruction and get high
nDCGscores,thus,theydon’tevaluateinstruction- API Models We use three of the best perform-
following. Hence why we propose a new metric ingAPIembeddingmodels: Cohere’sv3English,
disentangledfrommodel’skeyword-searchability. Google’s Gecko (Lee et al., 2024) and OpenAI’s
Text-Embedding-v3-Large. Itismostlyunknown
5Notethatwedonotshowstandardretrievalresultson
what these models’ training procedures were—
themodifiedinstruction’srelevantdocumentset,asstandard
including if they were trained on instructions or
retrievalscorescannotbedirectlycomparedacrossdifferent
queryrelevanceannotations(qrels). not—thus we place them in a distinct category.
11930

|     |            | Robust04  | News21     | Core17    | Average     |      |
| --- | ---------- | --------- | ---------- | --------- | ----------- | ---- |
|     | Model      | MAP p-MRR | nDCG p-MRR | MAP p-MRR | Score p-MRR |      |
|     | E5-base-v2 | 13.4 -6.7 | 20.9 -2.0  | 14.0 -2.9 | 16.1        | -3.9 |
RInoitcurtsnI-oN Contriever 19.7 -6.1 22.9 -2.8 15.3 -2.5 19.3 -3.8
|     | MonoBERT        | 21.0 -9.4  | 25.1 -0.8 | 18.4 -0.2 | 21.5 | -3.5 |
| --- | --------------- | ---------- | --------- | --------- | ---- | ---- |
|     | BM25            | 12.1 -3.1  | 19.3 -2.1 | 8.1 -1.1  | 13.2 | -2.1 |
|     | MonoT5-base     | 15.7 -6.2  | 11.0 +5.0 | 12.2 -4.1 | 13.0 | -1.8 |
|     | E5-large-v2     | 17.4 -4.2  | 24.3 +0.9 | 17.0 +0.1 | 19.6 | -1.1 |
|     | MonoT5-3B       | 27.3 +4.0  | 16.5 +1.8 | 18.2 +1.8 | 20.7 | +2.5 |
|     | TART-Contriever | 14.3 -9.0  | 21.8 -3.0 | 13.3 -3.0 | 16.5 | -5.0 |
|     | INSTRUCTOR-base | 17.2 -10.4 | 22.1 -1.8 | 15.5 -1.1 | 18.3 | -4.4 |
RI-noitcurtsnI
|     | E5-mistral           | 23.1 -9.6 | 27.8 -0.9 | 18.3 +0.1 | 23.1 | -3.5 |
| --- | -------------------- | --------- | --------- | --------- | ---- | ---- |
|     | BGE-base             | 16.8 -6.5 | 20.0 -0.1 | 14.6 -2.7 | 17.1 | -3.1 |
|     | INSTRUCTOR-xl        | 19.7 -8.1 | 26.1 -0.9 | 16.8 +0.7 | 20.9 | -2.8 |
|     | BGE-large            | 17.5 -7.8 | 22.3 +0.6 | 15.0 +0.1 | 18.3 | -2.4 |
|     | GritLM-7B            | 28.6 -1.7 | 24.4 -1.0 | 20.8 +2.6 | 24.6 | -0.0 |
|     | TART-FLAN-T5-xl      | 24.6 -0.7 | 12.8 +2.0 | 17.0 +2.8 | 18.1 | +1.4 |
|     | OpenAIv3Large        | 27.2 -5.8 | 27.2 -2.0 | 21.6 -0.2 | 25.3 | -2.7 |
|     | sIPA Coherev3English | 22.3 -3.6 | 28.3 +0.2 | 20.6 +2.8 | 23.7 | -0.2 |
|     | GoogleGecko          | 23.3 -2.4 | 29.5 +3.9 | 23.2 +5.4 | 25.3 | +2.3 |
|     | FLAN-T5-base         | 6.4 +5.3  | 6.1 -0.1  | 6.5 -3.3  | 6.3  | +0.6 |
sMLtcurtsnI Llama-2-7B-chat 6.3 +2.0 1.7 +0.2 5.4 +2.8 4.5 +1.7
|     | FLAN-T5-large   | 14.7 +3.9 | 8.0 +8.9  | 11.4 +1.3 | 11.4 | +4.7 |
| --- | --------------- | --------- | --------- | --------- | ---- | ---- |
|     | GritLM-Reranker | 9.7 +6.1  | 10.2 +3.4 | 9.8 +8.6  | 9.9  | +6.0 |
Mistral-7B-instruct 23.2 +12.6 27.2 +4.8 19.7 +13.0 23.4 +10.1
|     | FollowIR-7B | 24.8 +13.7 | 29.6 +6.3 | 20.0 +16.5 | 24.8 +12.2 |     |
| --- | ----------- | ---------- | --------- | ---------- | ---------- | --- |
Table2: Evaluatinginstruction-followingonFOLLOWIR.Introducedinthiswork,p-MRRisapairwiseevaluation
metricmeasuringinstructionfollowingwheninstructionschange, rangingfrom 100to100(higherisbetter).
−
Generallyonlymodelswithover3Bparametersorinstruction-tunedLMsthathaven’tbeentrainedonretrieval
tasksshowsuccessatfollowingretrievalinstruction.
However,wenotethatGoogle’smodeldidexplic- exception being GritLM (with scores averaging
itly train with instructions, as mentioned in their roughly zero) and TART-FLAN-T5-xl which has
technicalreport. slightlypositivescoresfortwoofthethreedatasets
(withanaverageof+1.4p-MRR).
| Instruction-Tuned | LMs | We also evaluate | sev- |     |     |     |
| ----------------- | --- | ---------------- | ---- | --- | --- | --- |
eralinstruction-tunedLMstobeusedasrerankers, APIModels WeseethattheAPImodelsperform
includingFLAN-T5(Chungetal.,2022),Llamav2 stronglyintermsofstandardIRmetrics,withOpe-
(Touvronetal.,2023b),andMistral-Instruct-v0.2 nAI’sandGoogle’smodelsperformingthehighest
(Jiang et al., 2023). We evaluate these models in overall. However,Cohere’sandOpenAI’smodels
thesamefashionasMonoT5rerankers,comparing performpoorlyatinstruction-followingwithneg-
the true and false tokens. Note that these models ative scores ( 0.2 and 2.7 on average, respec-
|     |     |     |     | −   | −   |     |
| --- | --- | --- | --- | --- | --- | --- |
werenottrainedonanyretrieval-specificdata. tively)whereasGoogleGeckohaspositivescores
(+2.3)likelyfromitsdatasetofinstructions.
| 4.2 FOLLOWIR | Results |     |                   |     |                         |     |
| ------------ | ------- | --- | ----------------- | --- | ----------------------- | --- |
|              |         |     | Instruct-TunedLMs |     | Incontrasttotheprevious |     |
Table2showsthemainresults,withthestandard
|     |     |     | results, | all instruction-tuned | LMs | show positive |
| --- | --- | --- | -------- | --------------------- | --- | ------------- |
IRscoreshown(eitherMAPornDCG@5)aswell
resultsforinstructionfollowing,althoughtheyhave
asthepairwiseevaluationmetric,p-MRR. thewidestrangeofperformanceusingstandardIR
No-InstructionIRModels Weseethattheno- metrics(rangingfromverypoorscorestosomeof
instructionmodelsrangewidelyinstandardIRmet- thehigherscores). Weseethatthebestperforming
rics(intermsofnDCG@5andMAP)butgenerally model in this category is FOLLOWIR–7B, which
havenegativescoresforp-MRR(upto 3.9). The wedescribeinmoredetailinSection5.
−
onlynon-instructionmodeltoscorepositivelyon
|     |     |     | Overall | We see that | the only models | that show |
| --- | --- | --- | ------- | ----------- | --------------- | --------- |
averageisMonoT5-3B(+2.5p-MRR).
positiveresultsatfollowinginstructionsareeither
InstructionIRModels Weagainseethatthese IR models with over 3B parameters or those that
models have generally negative scores, with the havebeenexplicitlytrainedtofollowinstructions
11931

No Instruction Training
0
5
10
MonoBERT E5-base-v2 MonoT5-base Contriever E5-large-v2 BM25 MonoT5-3B
5.0
2.5
0.0
2.5
E5-mistral INSTRUCTOR-b BGE-base TART-dual BGE-large INSTRUCTOR-xl GritLM-7B TART-FLAN
ylno
yreuq
eht
gnisu
ot
derapmoc
RRM-p
Uses Instructions in Training
Instruct-Tuned LLMs and APIs
5
0
5
OpenAI v3 Cohere v3 Google Gecko FLAN-T5-large GritLM-Reranker Mistral-7B-instruct FollowIR-7B
Instruction Setting
Keywords Short Instruction Full Instruction
Figure3: Scoredifferencebetweenusingnoinstructionstousinginstructionsformattedaskeywords,shorttext,or
thefulltext. Whilemodelsthatcancorrectlyuseinstructionsseegainswiththeadditionalinformation,mostother
modelsseedecreasingperformanceasinstructionlengthincreases.
(e.g. FLAN-T5), without any retrieval-specific authorsofthemodel(forBEIRdata). Forthefull
supervision. This aligns with work in the nat- prompttext,pleaseseeAppendixH.
ural language processing community which has
We show results for these ablations in Table 3,
shown that the instruction-following ability im-
where positive scores indicate that adding infor-
proveswithscale(Brownetal.,2020)andsuper-
mationimprovesthemodelwhilenegativescores
visedinstruction-tuning(Longpreetal.,2023).
indicateadropinperformance. Weseeaconsistent
trend where models that did poorly on longer in-
structionsperformbetteronkeywordsandshorter
4.3 Analysis
instructionsthanwiththefullinstruction. However,
modelsthatareabletofollowinstructionsgenerally
Whydosomanymodelsfailtocorrectlyfollowin-
seebetterresultswiththeadditionalinformation.
structionswhentheydowellontypicalIRmetrics
suchasnDCGandMAP?Weanswerthisquestion These results show that models are (1) using
by ablating several components that may impact theinstructiontextaskeywords(asperformanceis
results: (1)whetherIRmodelsarenotusedtotext higherwhenusingonlykeywords)and(2)arenot
thatcannotbeusedforsimplekeywordsearch(i.e. abletoutilizetheextrainformationintheinstruc-
instructions) and (2) whether they are unused to tions (as they generally decrease in performance
the length of the longer instructions (as current withthisadditionalinformation).
retrievershavebeentrainedonshorterinput).
We also confirm that these results hold on
To test these, we compare the original query- datasets outside of TREC collections and show
onlyresulttothosewhereweadditionallygivethe results on three BEIR datasets: SciFact, NFCor-
modeleitherthefullinstruction,ashorterinstruc- pus, and FiQA. We show in Table 3 the original
tion,orkeywordsfromtheinstruction. Wegather score (using the short instructions from their pa-
theseshortinstructionsandkeywordsbyprompt- pers)andthechangeinscorewhenusingjustkey-
ingGPT-4-Turbo-1106togeneratethemfromthe words from the instruction (again extracted from
originalfullinstruction(forTRECdata)orother- GPT-4). We show results only for models which
wiseusetheoriginalshortinstructionsgivenbythe performed poorly for instruction-following. We
11932

|     |     |     |       |     | SciFact    | NFCorpus  |         |      | FiQA       |     |     |
| --- | --- | --- | ----- | --- | ---------- | --------- | ------- | ---- | ---------- | --- | --- |
|     |     |     | Model |     | OG ∆w/Key. | OG        | ∆w/Key. |      | OG ∆w/Key. |     |     |
|     |     |     | BM25  |     | 67.9       | -1.7 32.2 |         | -5.1 | 23.6 -1.6  |     |     |
noitcurtsnI
|     |     |     | E5-base-v2 |     | 71.9 | -2.7 35.4 |     | -2.5 | 39.9 -0.4 |     |     |
| --- | --- | --- | ---------- | --- | ---- | --------- | --- | ---- | --------- | --- | --- |
-oN
|     |     |     | Contriever      |     | 64.9 | +0.4 31.7 | +0.0 |      | 24.5 -3.2 |     |     |
| --- | --- | --- | --------------- | --- | ---- | --------- | ---- | ---- | --------- | --- | --- |
|     |     |     | MonoT5-base     |     | 73.1 | -0.6 35.6 |      | -0.9 | 41.2 -0.3 |     |     |
|     |     |     | TART-Contriever |     | 67.6 | -0.3 33.4 |      | -5.3 | 31.8 -0.4 |     |     |
noitcurtsnIsesU
|     |     |     | INSTRUCTOR-base |     | 57.8 | +1.0 31.6 |      | -0.4 | 39.2 -0.1 |     |     |
| --- | --- | --- | --------------- | --- | ---- | --------- | ---- | ---- | --------- | --- | --- |
|     |     |     | BGE-base        |     | 73.2 | -0.5 35.5 | +0.0 |      | 40.8 -2.3 |     |     |
|     |     |     | TART-FLAN-xl    |     | 74.2 | +1.6 33.9 | +0.4 |      | 39.6 -0.3 |     |     |
|     |     |     | INSTRUCTOR-xl   |     | 62.4 | +0.2 36.0 |      | -0.6 | 46.9 +0.8 |     |     |
|     |     |     | E5-Mistral      |     | 77.1 | -5.1 38.8 | +0.3 |      | 56.7 -6.5 |     |     |
Table3: AblationonBEIRbenchmarksformodelsthatdopoorlywithlongerinstructions,comparingtheiroriginal
shortinstructionsvsdomainkeywordsextractedfromthoseinstructions(seeAppendixGforalist). OGstands
fororiginalinput. Ifmodelshadlearnedtousetheinstructionscorrectlywewouldseeadivergencebetweenthe
behaviorofinstructandnon-instructmodels,however,weseecomparableperformancebetweentheaddedkeywords
| vsthefullinstruction( |     |     | onepoint). |     |     |     |     |     |     |     |     |
| --------------------- | --- | --- | ---------- | --- | --- | --- | --- | --- | --- | --- | --- |
±
Model Robustness@10 The prompts for this experiment can be found in
AppendixH.
|     | BM25                |     |     | 26.9 |     |                                          |         |           |                      |     |             |
| --- | ------------------- | --- | --- | ---- | --- | ---------------------------------------- | ------- | --------- | -------------------- | --- | ----------- |
|     | TART-Contriever     |     |     | 47.5 |     |                                          |         |           |                      |     |             |
|     |                     |     |     |      |     | However,                                 | these   | synthetic | documents            |     | are noisy   |
|     | RepLLaMa            |     |     | 52.6 |     |                                          |         |           |                      |     |             |
|     |                     |     |     |      |     | and contains                             |         | errors    | w.r.t. the labels—to |     | remedy      |
|     | E5-Mistral          |     |     | 55.4 |     |                                          |         |           |                      |     |             |
|     |                     |     |     |      |     | this, we                                 | perform | a         | round of filtering   |     | and use the |
|     | Mistral-7B-instruct |     |     | 35.3 |     |                                          |         |           |                      |     |             |
|     | FollowIR-7B         |     |     | 71.5 |     | bestperformingopen-sourcemodelfromTable2 |         |           |                      |     |             |
(Mistral-7B-Instruct-v0.2)toscoreeachofthegen-
| Table | 4: Performance        |     | on the InstructIR | benchmark |         |                                           |     |     |     |     |     |
| ----- | --------------------- | --- | ----------------- | --------- | ------- | ----------------------------------------- | --- | --- | --- | --- | --- |
|       |                       |     |                   |           |         | erateddocumentsaccordingtotheinstruction. |     |     |     |     | We  |
| using | their “Robustness@10" |     | scores,           | e.g.      | the min |                                           |     |     |     |     |     |
thenfilterthedocumentsaccordingtowhetherMis-
| nDCG@10 | score | across | 10 instructions. |     | Upper por- |     |     |     |     |     |     |
| ------- | ----- | ------ | ---------------- | --- | ---------- | --- | --- | --- | --- | --- | --- |
tionisbi-encoderswhilelowerisrerankers. tral correctly predicts the generated label, and fi-
|     |     |     |     |     |     | nally balance |     | the relevant | and non-relevant |     | sam- |
| --- | --- | --- | --- | --- | --- | ------------- | --- | ------------ | ---------------- | --- | ---- |
see that the scores for keywords vs the short in- ples,choosingonlyonerelevantandnon-relevant
structionaregenerallysimilar,withmostmodels documentperquery. Ourtotalis 1800trainingin-
∼
|                       |     |     |                     |     |     | stanceson | 1200uniquequery/instructionspairs. |     |     |     |     |
| --------------------- | --- | --- | ------------------- | --- | --- | --------- | ---------------------------------- | --- | --- | --- | --- |
| seeingachangeofaround |     |     | 1point,exceptforthe |     |     |           |                                    |     |     |     |     |
|                       |     |     | ±                   |     |     |           | ∼                                  |     |     |     |     |
strongestofthenon-instruction-followingmodels, Wethentrainourinstruction-followingmodel,
E5-Mistral,seeingalargerdroponsomedatasets.
|         |     |      |             |           |     | FOLLOWIR-7B,                                |     |     | by fine-tuning |     | Mistral-7B- |
| ------- | --- | ---- | ----------- | --------- | --- | ------------------------------------------- | --- | --- | -------------- | --- | ----------- |
|         | We  | find | overall (on | both TREC | and |                                             |     |     |                |     |             |
| Overall |     |      |             |           |     | Instruct-v0.2onourdatausingtheLlama-Factory |     |     |                |     |             |
BEIR datasets) that models use instructions for framework(Hiyouga,2023)withLoRA(Huetal.,
keyword matching and are unused to longer in- 2021). Full training hyperparameter details are
| structionsthatmaycontainlessrelevantwords. |     |     |     |     |     | foundinAppendixC. |     |     |     |     |     |
| ------------------------------------------ | --- | --- | --- | --- | --- | ----------------- | --- | --- | --- | --- | --- |
5 TeachingInstructionFollowing When we evaluate this model on FOLLOWIR
|     |     |     |     |     |     | (Table | 2), we | find | that the scores |     | consistently |
| --- | --- | --- | --- | --- | --- | ------ | ------ | ---- | --------------- | --- | ------------ |
Isitpossibletoimprovemodelperformanceinfol-
|                     |     |     |                         |     |     | improve. | Compared |     | to the original |     | Mistral-7B- |
| ------------------- | --- | --- | ----------------------- | --- | --- | -------- | -------- | --- | --------------- | --- | ----------- |
| lowinginstructions? |     |     | Weshowthatfine-tuningon |     |     |          |          |     |                 |     |             |
Instruct-v0.2,ourmodelimprovesonbothstandard
a training set of longer instructions can provide IR metrics (+6.0% relative improvement) and on
a method for doing so. We start by gathering a instruction following (+20.8% relative). We also
| trainingsettoteachmodels. |     |     | WecollectallTREC |     |     |     |     |     |     |     |     |
| ------------------------- | --- | --- | ---------------- | --- | --- | --- | --- | --- | --- | --- | --- |
showthatthisimprovementholdsontheconcurrent
narratives(i.e.,instructions)fromtasksnotinFOL-
|     |     |     |     |     |     | InstructIR | dataset | (Table | 4), where | FollowIR-7B |     |
| --- | --- | --- | --- | --- | --- | ---------- | ------- | ------ | --------- | ----------- | --- |
LOWIR,consistingof1836pairsofqueriesandnar- scoresdoublethebaseMistral-7Bscores(71.5Ro-
ratives. However,wenotethatthisdoesnotprovide
bustness@10vs35.3)andisthetopscoringmodel
anypositiveornegativedocumentsforfine-tuning.
overall. Thus,wecanseethatitispossibletotrain
In order to obtain documents for training, we IRmodelstobebetterinstructionfollowers.6
promptGPT-3.5-Turbo-1106togeneraterelevant
| and not-relevant |     | documents, | generating |     | roughly |     |     |     |     |     |     |
| ---------------- | --- | ---------- | ---------- | --- | ------- | --- | --- | --- | --- | --- | --- |
tworelevantandnon-relevantinstancesperquery. 6See§BforaLlama-3-8Bbaseversion.
11933

| 6 Conclusion |     |     |     |     |     | 2017. | Trec2017commoncoretrackoverview. |     |     |     |     | In  |
| ------------ | --- | --- | --- | --- | --- | ----- | -------------------------------- | --- | --- | --- | --- | --- |
TREC.
DespitetheuseofLMsasthebackboneofneural
AkariAsai,TimoSchick,PatrickLewis,XilunChen,
| retrieval         | models, | most existing | IR       | models     | do not |                              |          |           |     |                     |     |     |
| ----------------- | ------- | ------------- | -------- | ---------- | ------ | ---------------------------- | -------- | --------- | --- | ------------------- | --- | --- |
|                   |         |               |          |            |        | Gautier                      | Izacard, | Sebastian |     | Riedel, Hannaneh    |     | Ha- |
| take instructions |         | that define   | document | relevance. |        |                              |          |           |     |                     |     |     |
|                   |         |               |          |            |        | jishirzi,andWen-tauYih.2022. |          |           |     | Task-awareretrieval |     |     |
Further,thereisnoexistingresourcethatmeasures withinstructions. arXivpreprintarXiv:2211.09260.
howwellretrievalmodelscanfollowinstructions.
|     |     |     |     |     |     | Yuntao Bai, | Andy | Jones, | Kamal | Ndousse, | Amanda |     |
| --- | --- | --- | --- | --- | --- | ----------- | ---- | ------ | ----- | -------- | ------ | --- |
Webuildanewbenchmarkthatexplicitlymeasures Askell, AnnaChen, NovaDasSarma, DawnDrain,
theinstructionfollowingabilityofretrievalmodels Stanislav Fort, Deep Ganguli, Tom Henighan,
andfindthatnearlyallretrievalmodelsdonotfol- NicholasJoseph,SauravKadavath,JacksonKernion,
TomConerly,SheerEl-Showk,NelsonElhage,Zac
lowinstructions,withtheexceptionoflargermod-
|     |     |     |     |     |     | Hatfield-Dodds, |     | Danny | Hernandez, | Tristan |     | Hume, |
| --- | --- | --- | --- | --- | --- | --------------- | --- | ----- | ---------- | ------- | --- | ----- |
els(3B+parameters)orinstruction-tunedLMsthat
ScottJohnston,ShaunaKravec,LianeLovitt,Neel
typically are not used for retrieval. However, we Nanda, Catherine Olsson, Dario Amodei, Tom
showthatitispossibletoimprovetheirinstruction Brown, Jack Clark, Sam McCandlish, Chris Olah,
|     |     |     |     |     |     | Ben Mann, |     | and Jared | Kaplan. | 2022. | Training |     |
| --- | --- | --- | --- | --- | --- | --------- | --- | --------- | ------- | ----- | -------- | --- |
followingability,andbuildandreleaseatraining
|             |                               |           |        |           |     | a helpful         | and      | harmless | assistant | with      | reinforce- |     |
| ----------- | ----------------------------- | --------- | ------ | --------- | --- | ----------------- | -------- | -------- | --------- | --------- | ---------- | --- |
| corpus for  | teaching                      | retrieval | models | to follow | in- |                   |          |          |           |           |            |     |
|             |                               |           |        |           |     | ment              | learning | from     | human     | feedback. | Preprint,  |     |
| structions. | Ournewmodel,FOLLOWIR-7B,shows |           |        |           |     | arXiv:2204.05862. |          |          |           |           |            |     |
improvementonbothstandardretrievalmetricsas
|         |                |            |     |          |      | Parishad | BehnamGhader, |     | Vaibhav | Adlakha, | Marius |     |
| ------- | -------------- | ---------- | --- | -------- | ---- | -------- | ------------- | --- | ------- | -------- | ------ | --- |
| well as | in instruction | following, |     | which we | hope |          |               |     |         |          |        |     |
Mosbach,DzmitryBahdanau,NicolasChapados,and
willinspirefutureworkonthetopic. SivaReddy.2024. Llm2vec: Largelanguagemodels
|               |     |     |     |     |     | aresecretlypowerfultextencoders. |     |     |     | arXivpreprint |     |     |
| ------------- | --- | --- | --- | --- | --- | -------------------------------- | --- | --- | --- | ------------- | --- | --- |
| 7 Limitations |     |     |     |     |     | arXiv:2404.05961.                |     |     |     |               |     |     |
FedericoBianchi,MiracSuzgun,GiuseppeAttanasio,
| Reranking  | vs Full     | Retrieval |     | As our setup | for      |                   |     |     |                     |           |            |     |
| ---------- | ----------- | --------- | --- | ------------ | -------- | ----------------- | --- | --- | ------------------- | --------- | ---------- | --- |
|            |             |           |     |              |          | Paul Röttger,     |     | Dan | Jurafsky,           | Tatsunori | Hashimoto, |     |
| evaluating | instruction | following |     | requires     | evaluat- |                   |     |     |                     |           |            |     |
|            |             |           |     |              |          | andJamesZou.2024. |     |     | Safety-tunedllamas: |           | Lessons    |     |
ing the documents which changed relevance, we fromimprovingthesafetyoflargelanguagemodels
cannotusethefullcollectionforretrieval(aseach thatfollowinstructions. Preprint,arXiv:2309.07875.
retrieverfindsdifferentrelevantdocumentsbyde-
SidBlack,StellaBiderman,EricHallahan,QuentinG.
sign). Further, due to licensing restrictions, we Anthony, Leo Gao, Laurence Golding, Horace
cannotdistributethefullcorporafromtheTREC He, Connor Leahy, Kyle McDonell, Jason Phang,
MichaelMartinPieler,USVSNSaiPrashanth,Shiv-
tracks—thuswedistributepassagesduetofairuse
anshuPurohit,LariaReynolds,JonathanTow,Benqi
| laws. However, |     | we show | full corpus | retrieval | re- |       |     |        |           |       |           |     |
| -------------- | --- | ------- | ----------- | --------- | --- | ----- | --- | ------ | --------- | ----- | --------- | --- |
|                |     |         |             |           |     | Wang, | and | Samuel | Weinbach. | 2022. | Gpt-neox- |     |
sultsforasubsetofmodelsinAppendixFandnote 20b: Anopen-sourceautoregressivelanguagemodel.
similar trends in terms of the lack of instruction ArXiv,abs/2204.06745.
following.
|     |     |     |     |     |     | Tom Brown, | Benjamin |     | Mann, | Nick Ryder, | Melanie |     |
| --- | --- | --- | --- | --- | --- | ---------- | -------- | --- | ----- | ----------- | ------- | --- |
PossibleErrors OurworkisbuiltontheTREC Subbiah,JaredDKaplan,PrafullaDhariwal,Arvind
Neelakantan,PranavShyam,GirishSastry,Amanda
| document       | collections | and      | judgements,  | as    | well as |                   |                                       |     |                           |     |     |     |
| -------------- | ----------- | -------- | ------------ | ----- | ------- | ----------------- | ------------------------------------- | --- | ------------------------- | --- | --- | --- |
|                |             |          |              |       |         | Askell,etal.2020. |                                       |     | Languagemodelsarefew-shot |     |     |     |
| new annotation |             | efforts. | We do not    | check | for po- |                   |                                       |     |                           |     |     |     |
|                |             |          |              |       |         | learners.         | Advancesinneuralinformationprocessing |     |                           |     |     |     |
| tential errors | in          | the TREC | annotations, | and   | our     |                   |                                       |     |                           |     |     |     |
systems,33:1877–1901.
newlygatheredannotationsmayhavesmallerrors.
|     |     |     |     |     |     | Yinqiong | Cai, | Jiafeng | Guo, Yixing | Fan, | Qingyao | Ai, |
| --- | --- | --- | --- | --- | --- | -------- | ---- | ------- | ----------- | ---- | ------- | --- |
Despitethesecaveats,weseethatourdatasetstill
|     |     |     |     |     |     | RuqingZhang,andXueqiCheng.2022. |     |     |     |     | Hardneg- |     |
| --- | --- | --- | --- | --- | --- | ------------------------------- | --- | --- | --- | --- | -------- | --- |
provides a useful evaluation setup for measuring ativesorfalsenegatives: Correctingpoolingbiasin
instructionfollowing. trainingneuralrankingmodels. InProceedingsofthe
31stACMInternationalConferenceonInformation
&KnowledgeManagement,pages118–127.
8 Acknowledgments
HaonanChen,ZhichengDou,KelongMao,Jiongnan
OWissupportedbyaNSFGRFPfellowship.
|     |     |     |     |     |     | Liu,andZiliangZhao.2024. |     |     |     | Generalizingconversa- |     |     |
| --- | --- | --- | --- | --- | --- | ------------------------ | --- | --- | --- | --------------------- | --- | --- |
tionaldenseretrievalviallm-cognitiondataaugmen-
tation. arXivpreprintarXiv:2402.07092.
References
JianlvChen,ShitaoXiao,PeitianZhang,KunLuo,Defu
AI@Meta.2024. Llama3modelcard. Lian, and Zheng Liu. 2023. Bge m3-embedding:
Multi-lingual,multi-functionality,multi-granularity
JamesAllan,DonnaHarman,EvangelosKanoulas,Dan textembeddingsthroughself-knowledgedistillation.
Li, Christophe Van Gysel, and Ellen M Voorhees. Preprint,arXiv:2309.07597.
11934

Mark Chen, Jerry Tworek, Heewoo Jun, Qiming GautierIzacard,MathildeCaron,LucasHosseini,Se-
Yuan,HenriquePondedeOliveiraPinto,JaredKa- bastian Riedel, Piotr Bojanowski, Armand Joulin,
plan, HarriEdwards, YuriBurda, NicholasJoseph, andEdouardGrave.2021. Unsuperviseddensein-
arXiv
Greg Brockman, et al. 2021. Evaluating large formationretrievalwithcontrastivelearning.
language models trained on code. arXiv preprint preprintarXiv:2112.09118.
arXiv:2107.03374.
AlbertQiaochuJiang,AlexandreSablayrolles,Arthur
Wei-LinChiang,LianminZheng,YingSheng,Anasta- Mensch, Chris Bamford, Devendra Singh Chap-
siosNikolasAngelopoulos,TianleLi,DachengLi, lot, Diego de Las Casas, Florian Bressand, Gi-
|     |     |     |     |     |     | annaLengyel, |     | GuillaumeLample, |     |     | LucileSaulnier, |     |
| --- | --- | --- | --- | --- | --- | ------------ | --- | ---------------- | --- | --- | --------------- | --- |
HaoZhang,BanghuaZhu,MichaelJordan,JosephE.
Gonzalez,andIonStoica.2024. Chatbotarena: An L’elioRenardLavaud,Marie-AnneLachaux,Pierre
openplatformforevaluatingllmsbyhumanprefer- Stock,TevenLeScao,ThibautLavril,ThomasWang,
ence. Preprint,arXiv:2403.04132. TimothéeLacroix,andWilliamElSayed.2023. Mis-
|     |     |     |     |     |     | tral7b. | ArXiv,abs/2310.06825. |     |     |     |     |     |
| --- | --- | --- | --- | --- | --- | ------- | --------------------- | --- | --- | --- | --- | --- |
HyungWonChung,LeHou,ShayneLongpre,Barret
|     |     |     |     |     |     | Carlos E | Jimenez, | John | Yang, | Alexander |     | Wettig, |
| --- | --- | --- | --- | --- | --- | -------- | -------- | ---- | ----- | --------- | --- | ------- |
Zoph,YiTay,WilliamFedus,YunxuanLi,Xuezhi
|               |           |     |            |     |             | Shunyu | Yao, Kexin | Pei, | Ofir | Press, | and Karthik | R   |
| ------------- | --------- | --- | ---------- | --- | ----------- | ------ | ---------- | ---- | ---- | ------ | ----------- | --- |
| Wang, Mostafa | Dehghani, |     | Siddhartha |     | Brahma, Al- |        |            |      |      |        |             |     |
bert Webson, Shixiang Shane Gu, Zhuyun Dai, Narasimhan.2024. SWE-bench: Canlanguagemod-
MiracSuzgun,XinyunChen,AakankshaChowdh- elsresolvereal-worldgithubissues? InTheTwelfth
|     |     |     |     |     |     | International |     | Conference | on  | Learning | Representa- |     |
| --- | --- | --- | --- | --- | --- | ------------- | --- | ---------- | --- | -------- | ----------- | --- |
ery,AlexCastro-Ros,MariePellat,KevinRobinson,
tions.
DashaValter,SharanNarang,GauravMishra,Adams
| Yu, Vincent | Zhao, | Yanping | Huang, |     | Andrew Dai, |     |     |     |     |     |     |     |
| ----------- | ----- | ------- | ------ | --- | ----------- | --- | --- | --- | --- | --- | --- | --- |
VladimirKarpukhin,BarlasOg˘uz,SewonMin,Patrick
HongkunYu,SlavPetrov,EdH.Chi,JeffDean,Ja-
cobDevlin,AdamRoberts,DennyZhou,QuocV.Le, Lewis,LedellWu,SergeyEdunov,DanqiChen,and
|                   |     |                              |     |     |     | Wen-tau     | Yih. | 2020.    | Dense      | passage | retrieval | for      |
| ----------------- | --- | ---------------------------- | --- | --- | --- | ----------- | ---- | -------- | ---------- | ------- | --------- | -------- |
| andJasonWei.2022. |     | Scalinginstruction-finetuned |     |     |     |             |      |          |            |         |           |          |
|                   |     |                              |     |     |     | open-domain |      | question | answering. |         | arXiv     | preprint |
| languagemodels.   |     | Preprint,arXiv:2210.11416.   |     |     |     |             |      |          |            |         |           |          |
arXiv:2004.04906.
| ZhuyunDaiandJamieCallan.2019. |        |                 |     | Deepertextun- |          |                                  |     |     |     |     |          |       |
| ----------------------------- | ------ | --------------- | --- | ------------- | -------- | -------------------------------- | --- | --- | --- | --- | -------- | ----- |
|                               |        |                 |     |               |          | OmarKhattabandMateiZaharia.2020. |     |     |     |     | Colbert: | Effi- |
| derstanding                   | for ir | with contextual |     | neural        | language |                                  |     |     |     |     |          |       |
cientandeffectivepassagesearchviacontextualized
| modeling. | InProceedingsofthe42ndinternational |     |     |     |     |                          |     |     |                        |     |     |     |
| --------- | ----------------------------------- | --- | --- | --- | --- | ------------------------ | --- | --- | ---------------------- | --- | --- | --- |
|           |                                     |     |     |     |     | lateinteractionoverbert. |     |     | InProceedingsofthe43rd |     |     |     |
ACMSIGIRconferenceonresearchanddevelopment
|     |     |     |     |     |     | International |     | ACM SIGIR | conference |     | on  | research |
| --- | --- | --- | --- | --- | --- | ------------- | --- | --------- | ---------- | --- | --- | -------- |
ininformationretrieval,pages985–988.
anddevelopmentinInformationRetrieval,pages39–
48.
DirkGroeneveld,IzBeltagy,PeteWalsh,AkshitaBha-
gia,RodneyKinney,OyvindTafjord,A.Jha,Hamish DawnLawrie,SeanMacAvaney,JamesMayfield,Paul
Ivison,IanMagnusson,YizhongWang,ShaneArora,
|                 |       |         |          |         |         | McNamee, | Douglas |       | W. Oard, | Luca | Soldaini, | and  |
| --------------- | ----- | ------- | -------- | ------- | ------- | -------- | ------- | ----- | -------- | ---- | --------- | ---- |
| David Atkinson, |       | Russell | Authur,  | Khyathi | Raghavi |          |         |       |          |      |           |      |
|                 |       |         |          |         |         | Eugene   | Yang.   | 2024. | Overview | of   | the TREC  | 2023 |
| Chandu,         | Arman | Cohan,  | Jennifer | Dumas,  | Yanai   |          |         |       |          |      |           |      |
NeuCLIRtrack.
Elazar,YulingGu,JackHessel,TusharKhot,William
Merrill,JacobDanielMorrison,NiklasMuennighoff,
|     |     |     |     |     |     | Jinhyuk Lee, | Zhuyun | Dai, | Xiaoqi | Ren, | Blair | Chen, |
| --- | --- | --- | --- | --- | --- | ------------ | ------ | ---- | ------ | ---- | ----- | ----- |
AakankshaNaik, CrystalNam, MatthewE.Peters, Daniel Cer, Jeremy R Cole, Kai Hui, Michael Bo-
Valentina Pyatkin, Abhilasha Ravichander, Dustin ratko,RajviKapadia,WenDing,etal.2024. Gecko:
Schwenk,SaurabhShah,WillSmith,EmmaStrubell,
|         |            |          |           |             |         | Versatile    | text | embeddings                     | distilled |     | from | large lan- |
| ------- | ---------- | -------- | --------- | ----------- | ------- | ------------ | ---- | ------------------------------ | --------- | --- | ---- | ---------- |
| Nishant | Subramani, | Mitchell | Wortsman, |             | Pradeep |              |      |                                |           |     |      |            |
|         |            |          |           |             |         | guagemodels. |      | arXivpreprintarXiv:2403.20327. |           |     |      |            |
| Dasigi, | Nathan     | Lambert, | Kyle      | Richardson, | Luke    |              |      |                                |           |     |      |            |
Zettlemoyer,JesseDodge,KyleLo,LucaSoldaini, XianmingLiandJingLi.2023. Angle-optimizedtext
NoahA.Smith,andHannaHajishirzi.2024. Olmo: embeddings. arXivpreprintarXiv:2309.12871.
Acceleratingthescienceoflanguagemodels.
|     |     |     |     |     |     | Percy Liang, | Rishi | Bommasani, |     | Tony | Lee, | Dimitris |
| --- | --- | --- | --- | --- | --- | ------------ | ----- | ---------- | --- | ---- | ---- | -------- |
Hiyouga.2023. Llamafactory. https://github.com/ Tsipras, Dilara Soylu, Michihiro Yasunaga, Yian
hiyouga/LLaMA-Factory.
|     |     |     |     |     |     | Zhang, | Deepak | Narayanan, |     | Yuhuai | Wu, | Ananya |
| --- | --- | --- | --- | --- | --- | ------ | ------ | ---------- | --- | ------ | --- | ------ |
Kumar,BenjaminNewman,BinhangYuan,Bobby
Edward J Hu, Yelong Shen, Phillip Wallis, Zeyuan Yan,CeZhang,ChristianCosgrove,ChristopherD.
Allen-Zhu, Yuanzhi Li, Shean Wang, Lu Wang, Manning, Christopher Ré, Diana Acosta-Navas,
and Weizhu Chen. 2021. Lora: Low-rank adap- Drew A. Hudson, Eric Zelikman, Esin Durmus,
tation of large language models. arXiv preprint FaisalLadhak, FriedaRong, HongyuRen, Huaxiu
arXiv:2106.09685. Yao, Jue Wang, Keshav Santhanam, Laurel Orr,
|     |     |     |     |     |     | Lucia | Zheng, | Mert | Yuksekgonul, |     | Mirac | Suzgun, |
| --- | --- | --- | --- | --- | --- | ----- | ------ | ---- | ------------ | --- | ----- | ------- |
Hamish Ivison, Yizhong Wang, Valentina Pyatkin, Nathan Kim, Neel Guha, Niladri Chatterji, Omar
Nathan Lambert, Matthew Peters, Pradeep Dasigi, Khattab, PeterHenderson, QianHuang, RyanChi,
Joel Jang, David Wadden, Noah A. Smith, Iz Belt- Sang Michael Xie, Shibani Santurkar, Surya Gan-
agy, and Hannaneh Hajishirzi. 2023. Camels in a guli, Tatsunori Hashimoto, Thomas Icard, Tianyi
changingclimate: Enhancinglmadaptationwithtulu Zhang,VishravChaudhary,WilliamWang,Xuechen
2. Preprint,arXiv:2311.10702. Li,YifanMai,YuhuiZhang,andYutaKoreeda.2023.
11935

Holistic evaluation of language models. Preprint, LongOuyang,JeffreyWu,XuJiang,DiogoAlmeida,
arXiv:2211.09110. CarrollWainwright,PamelaMishkin,ChongZhang,
SandhiniAgarwal,KatarinaSlama,AlexRay,etal.
| Shayne | Longpre, | Le     | Hou, Tu | Vu, Albert  | Webson, |            |                                        |           |          |     |           |
| ------ | -------- | ------ | ------- | ----------- | ------- | ---------- | -------------------------------------- | --------- | -------- | --- | --------- |
|        |          |        |         |             |         | 2022b.     | Traininglanguagemodelstofollowinstruc- |           |          |     |           |
| Hyung  | Won      | Chung, | Yi Tay, | Denny Zhou, | Quoc V  |            |                                        |           |          |     |           |
|        |          |        |         |             |         | tions with | human                                  | feedback. | Advances |     | in Neural |
Le, Barret Zoph, Jason Wei, et al. 2023. The flan InformationProcessingSystems,35:27730–27744.
| collection: | Designingdataandmethodsforeffective |     |     |     |     |     |     |     |     |     |     |
| ----------- | ----------------------------------- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
instructiontuning. InInternationalConferenceon Piotr Padlewski, Max Bain, Matthew Henderson,
MachineLearning,pages22631–22648.PMLR. ZhongkaiZhu,NishantRelan,HaiPham,Donovan
|     |     |     |     |     |     | Ong, Kaloyan |     | Aleksiev, | Aitor | Ormazabal, | Samuel |
| --- | --- | --- | --- | --- | --- | ------------ | --- | --------- | ----- | ---------- | ------ |
XueguangMa,LiangWang,NanYang,FuruWei,and
|                |     |                                |     |     |     | Phua,etal.2024. |     | Vibe-eval: | Ahardevaluationsuite |     |     |
| -------------- | --- | ------------------------------ | --- | --- | --- | --------------- | --- | ---------- | -------------------- | --- | --- |
| JimmyLin.2023. |     | Fine-tuningllamaformulti-stage |     |     |     |                 |     |            |                      |     |     |
formeasuringprogressofmultimodallanguagemod-
arXivpreprintarXiv:2310.08319.
| textretrieval. |     |     |     |     |     | els. arXivpreprintarXiv:2405.02287. |     |     |     |     |     |
| -------------- | --- | --- | --- | --- | --- | ----------------------------------- | --- | --- | --- | --- | --- |
Niklas Muennighoff, Hongjin Su, Liang Wang, Nan RonakPradeep,SahelSharifymoghaddam,andJimmy
Yang, Furu Wei, Tao Yu, Amanpreet Singh, and Lin.2023. Rankzephyr: Effectiveandrobustzero-
DouweKiela.2024. Generativerepresentationalin- shotlistwisererankingisabreeze! arXivpreprint
| structiontuning. |     | arXivpreprintarXiv:2402.09906. |     |     |     |     |     |     |     |     |     |
| ---------------- | --- | ------------------------------ | --- | --- | --- | --- | --- | --- | --- | --- | --- |
arXiv:2312.02724.
NiklasMuennighoff,NouamaneTazi,LoïcMagne,and
YujiaQin,ShihaoLiang,YiningYe,KunlunZhu,Lan
| NilsReimers.2022. |     |     | Mteb: Massivetextembedding |     |     |     |     |     |     |     |     |
| ----------------- | --- | --- | -------------------------- | --- | --- | --- | --- | --- | --- | --- | --- |
Yan,YaxiLu,YankaiLin,XinCong,XiangruTang,
benchmark. arXivpreprintarXiv:2210.07316. BillQian,SihanZhao,LaurenHong,RunchuTian,
|                                         |         |                |        |           |              | Ruobing                        | Xie,  | Jie Zhou,                  | Mark   | Gerstein, | Dahai Li,    |
| --------------------------------------- | ------- | -------------- | ------ | --------- | ------------ | ------------------------------ | ----- | -------------------------- | ------ | --------- | ------------ |
| Tri Nguyen,                             |         | Mir Rosenberg, |        | Xia Song, | Jianfeng     |                                |       |                            |        |           |              |
|                                         |         |                |        |           |              | ZhiyuanLiu,andMaosongSun.2023. |       |                            |        |           | Toolllm: Fa- |
| Gao,                                    | Saurabh | Tiwary,        | Rangan | Majumder, | and          |                                |       |                            |        |           |              |
|                                         |         |                |        |           |              | cilitating                     | large | language                   | models | to master | 16000+       |
| Li Deng.                                | 2016.   | MS             | MARCO: | A         | human gener- |                                |       |                            |        |           |              |
|                                         |         |                |        |           |              | real-worldapis.                |       | Preprint,arXiv:2307.16789. |        |           |              |
| atedmachinereadingcomprehensiondataset. |         |                |        |           | CoRR,        |                                |       |                            |        |           |              |
abs/1611.09268.
RafaelRafailov,ArchitSharma,EricMitchell,Christo-
pherDManning,StefanoErmon,andChelseaFinn.
| Ansong | Ni, Matt | Gardner, | and | Pradeep | Dasigi. 2021. |     |     |     |     |     |     |
| ------ | -------- | -------- | --- | ------- | ------------- | --- | --- | --- | --- | --- | --- |
2023. Directpreferenceoptimization:Yourlanguage
Mitigatingfalse-negativecontextsinmulti-document
|          |           |     |                |                  |     | model | is secretly | a reward | model. | In  | Advances in |
| -------- | --------- | --- | -------------- | ---------------- | --- | ----- | ----------- | -------- | ------ | --- | ----------- |
| question | answering |     | with retrieval | marginalization. |     |       |             |          |        |     |             |
NeuralInformationProcessingSystems,volume36,
arXivpreprintarXiv:2103.12235.
pages53728–53741.CurranAssociates,Inc.
| Rodrigo | Nogueira   | and  | Kyunghyun | Cho.  | 2019. Pas- |                                   |     |     |     |                |     |
| ------- | ---------- | ---- | --------- | ----- | ---------- | --------------------------------- | --- | --- | --- | -------------- | --- |
|         |            |      |           |       |            | NilsReimersandIrynaGurevych.2019. |     |     |     | Sentence-bert: |     |
| sage    | re-ranking | with | bert.     | arXiv | preprint   |                                   |     |     |     |                |     |
Sentenceembeddingsusingsiamesebert-networks.
arXiv:1901.04085.
arXivpreprintarXiv:1908.10084.
RodrigoNogueira,ZhiyingJiang,andJimmyLin.2020.
|          |     |         |                   |     |              | Stephen | E Robertson, | Steve | Walker, | Susan | Jones, |
| -------- | --- | ------- | ----------------- | --- | ------------ | ------- | ------------ | ----- | ------- | ----- | ------ |
| Document |     | ranking | with a pretrained |     | sequence-to- |         |              |       |         |       |        |
MichelineMHancock-Beaulieu,MikeGatford,etal.
| sequencemodel. |     | arXivpreprintarXiv:2003.06713. |     |     |     |       |                |                           |     |     |     |
| -------------- | --- | ------------------------------ | --- | --- | --- | ----- | -------------- | ------------------------- | --- | --- | --- |
|                |     |                                |     |     |     | 1995. | Okapiattrec-3. | NistSpecialPublicationSp, |     |     |     |
109:109.
| Rodrigo   | Nogueira,                      | Wei   | Yang,       | Kyunghyun | Cho, and |              |         |           |       |           |         |
| --------- | ------------------------------ | ----- | ----------- | --------- | -------- | ------------ | ------- | --------- | ----- | --------- | ------- |
| Jimmy     | Lin.                           | 2019. | Multi-stage | document  | ranking  |              |         |           |       |           |         |
|           |                                |       |             |           |          | Victor Sanh, | Albert  | Webson,   | Colin | Raffel,   | Stephen |
| withbert. | arXivpreprintarXiv:1910.14424. |       |             |           |          |              |         |           |       |           |         |
|           |                                |       |             |           |          | Bach,        | Lintang | Sutawika, | Zaid  | Alyafeai, | Antoine |
DouglasWOard,BjörnHedin,StephenTomlinson,and Chaffin, Arnaud Stiegler, Arun Raja, Manan Dey,
Jason R Baron. 2008. Overview of the trec 2008 M Saiful Bari, Canwen Xu, Urmish Thakker,
ShanyaSharmaSharma,ElizaSzczechla,Taewoon
| legaltrack. |     | InTREC,pages500–277. |     |     |     |             |     |            |       |        |           |
| ----------- | --- | -------------------- | --- | --- | --- | ----------- | --- | ---------- | ----- | ------ | --------- |
|             |     |                      |     |     |     | Kim, Gunjan |     | Chhablani, | Nihal | Nayak, | Debajyoti |
HanseokOh,HyunjiLee,SeonghyeonYe,HaebinShin, Datta,JonathanChang,MikeTian-JianJiang,Han
Hansol Jang, Changwook Jun, and Minjoon Seo. Wang,MatteoManica,ShengShen,ZhengXinYong,
2024. Instructir: Abenchmarkforinstructionfollow- HarshitPandey,RachelBawden,ThomasWang,Tr-
ingofinformationretrievalmodels. arXivpreprint ishala Neeraj, Jos Rozen, Abheesht Sharma, An-
arXiv:2402.14334. dreaSantilli,ThibaultFevry,JasonAlanFries,Ryan
Teehan,TevenLeScao,StellaBiderman,LeoGao,
LongOuyang,JeffreyWu,XuJiang,DiogoAlmeida,
|     |     |     |     |     |     | ThomasWolf,andAlexanderMRush.2022. |     |     |     |     | Multi- |
| --- | --- | --- | --- | --- | --- | ---------------------------------- | --- | --- | --- | --- | ------ |
CarrollWainwright,PamelaMishkin,ChongZhang, taskpromptedtrainingenableszero-shottaskgener-
SandhiniAgarwal,KatarinaSlama,AlexRay,John alization. InInternationalConferenceonLearning
| Schulman,JacobHilton,FraserKelton,LukeMiller, |         |        |         |       |           | Representations. |     |     |     |     |     |
| --------------------------------------------- | ------- | ------ | ------- | ----- | --------- | ---------------- | --- | --- | --- | --- | --- |
| Maddie                                        | Simens, | Amanda | Askell, | Peter | Welinder, |                  |     |     |     |     |     |
PaulFChristiano,JanLeike,andRyanLowe.2022a. Shohreh Shaghaghian, Luna Yue Feng, Borna Jafar-
Traininglanguagemodelstofollowinstructionswith pour,andNicolaiPogrebnyakov.2020. Customizing
humanfeedback. InAdvancesinNeuralInformation contextualizedlanguagemodelsforlegaldocument
ProcessingSystems,volume35,pages27730–27744. reviews. In2020IEEEInternationalConferenceon
| CurranAssociates,Inc. |     |     |     |     |     | BigData(BigData),pages2139–2148.IEEE. |     |     |     |     |     |
| --------------------- | --- | --- | --- | --- | --- | ------------------------------------- | --- | --- | --- | --- | --- |
11936

KaranSinghal,ShekoofehAzizi,TaoTu,SSaraMah- andFuruWei.2022a. Textembeddingsbyweakly-
davi,JasonWei,HyungWonChung,NathanScales, supervisedcontrastivepre-training. arXivpreprint
| AjayTanwani,HeatherCole-Lewis,StephenPfohl, |                                   |     |     |     |     |     | arXiv:2212.03533. |     |     |     |     |     |     |
| ------------------------------------------- | --------------------------------- | --- | --- | --- | --- | --- | ----------------- | --- | --- | --- | --- | --- | --- |
| etal.2023.                                  | Largelanguagemodelsencodeclinical |     |     |     |     |     |                   |     |     |     |     |     |     |
knowledge. Nature,620(7972):172–180. LiangWang,NanYang,XiaolongHuang,LinjunYang,
|     |     |     |     |     |     |     | RanganMajumder,andFuruWei.2023a. |     |     |     |     | Improving |     |
| --- | --- | --- | --- | --- | --- | --- | -------------------------------- | --- | --- | --- | --- | --------- | --- |
IanSoboroff.2021. Overviewoftrec2021. In30thText textembeddingswithlargelanguagemodels. arXiv
REtrievalConference.Gaithersburg,Maryland. preprintarXiv:2401.00368.
| Ian Soboroff,      | Shudong   |      | Huang, | and       | Donna | Harman. |                                          |        |        |         |         |         |         |
| ------------------ | --------- | ---- | ------ | --------- | ----- | ------- | ---------------------------------------- | ------ | ------ | ------- | ------- | ------- | ------- |
|                    |           |      |        |           |       |         | Yizhong                                  | Wang,  | Hamish | Ivison, | Pradeep | Dasigi, | Jack    |
| 2018.              | Trec 2018 | news | track  | overview. | In    | TREC,   |                                          |        |        |         |         |         |         |
|                    |           |      |        |           |       |         | Hessel,                                  | Tushar | Khot,  | Khyathi | Raghavi |         | Chandu, |
| volume409,page410. |           |      |        |           |       |         | DavidWadden,KelseyMacMillan,NoahA.Smith, |        |        |         |         |         |         |
|                    |           |      |        |           |       |         | IzBeltagy,andHannanehHajishirzi.2023b.   |        |        |         |         |         | Howfar  |
Ian Soboroff, Shudong Huang, and Donna Harman. cancamelsgo? exploringthestateofinstructiontun-
| 2020. Trec2020newstrackoverview. |     |     |     |     | InTREC. |     |                     |     |     |                            |     |     |     |
| -------------------------------- | --- | --- | --- | --- | ------- | --- | ------------------- | --- | --- | -------------------------- | --- | --- | --- |
|                                  |     |     |     |     |         |     | ingonopenresources. |     |     | Preprint,arXiv:2306.04751. |     |     |     |
HongjinSu,WeijiaShi,JungoKasai,YizhongWang,
|           |      |            |     |         |      |         | YizhongWang, |     | YeganehKordi, |     | SwaroopMishra, |     | Al- |
| --------- | ---- | ---------- | --- | ------- | ---- | ------- | ------------ | --- | ------------- | --- | -------------- | --- | --- |
| Yushi Hu, | Mari | Ostendorf, |     | Wen-tau | Yih, | Noah A. |              |     |               |     |                |     |     |
isaLiu,NoahASmith,DanielKhashabi,andHan-
| Smith,    | Luke Zettlemoyer, |                             | and | Tao | Yu. 2022. | One      |                                          |     |     |                |     |              |       |
| --------- | ----------------- | --------------------------- | --- | --- | --------- | -------- | ---------------------------------------- | --- | --- | -------------- | --- | ------------ | ----- |
|           |                   |                             |     |     |           |          | nanehHajishirzi.2022b.                   |     |     | Self-instruct: |     | Aligninglan- |       |
| embedder, | any               | task: Instruction-finetuned |     |     |           | text em- |                                          |     |     |                |     |              |       |
|           |                   |                             |     |     |           |          | guagemodelwithselfgeneratedinstructions. |     |     |                |     |              | arXiv |
beddings.
preprintarXiv:2212.10560.
| Nandan Thakur,                          | Nils | Reimers,  |     | Andreas       | Rücklé,       | Ab-     |               |            |         |         |          |       |          |
| --------------------------------------- | ---- | --------- | --- | ------------- | ------------- | ------- | ------------- | ---------- | ------- | ------- | -------- | ----- | -------- |
|                                         |      |           |     |               |               |         | Yizhong       | Wang,      | Swaroop | Mishra, |          | Pegah | Alipoor- |
| hishekSrivastava,andIrynaGurevych.2021. |      |           |     |               |               | Beir:   |               |            |         |         |          |       |          |
|                                         |      |           |     |               |               |         | molabashi,    | Yeganeh    |         | Kordi,  | Amirreza |       | Mirzaei, |
| A heterogenous                          |      | benchmark |     | for zero-shot |               | evalua- |               |            |         |         |          |       |          |
|                                         |      |           |     |               |               |         | Anjana        | Arunkumar, |         | Arjun   | Ashok,   | Arut  | Selvan   |
| tionofinformationretrievalmodels.       |      |           |     |               | arXivpreprint |         |               |            |         |         |          |       |          |
|                                         |      |           |     |               |               |         | Dhanasekaran, |            | Atharva | Naik,   | David    | Stap, | et al.   |
arXiv:2104.08663.
|     |     |     |     |     |     |     | 2022c.      | Super-naturalinstructions: |     |     | Generalizationvia |            |       |
| --- | --- | --- | --- | --- | --- | --- | ----------- | -------------------------- | --- | --- | ----------------- | ---------- | ----- |
|     |     |     |     |     |     |     | declarative | instructions               |     | on  | 1600+             | nlp tasks. | arXiv |
HugoTouvron,ThibautLavril,GautierIzacard,Xavier
preprintarXiv:2204.07705.
Martinet,Marie-AnneLachaux,TimothéeLacroix,
BaptisteRozière,NamanGoyal,EricHambro,Faisal
Azhar, et al. 2023a. Llama: Open and effi- WilliamWebber,AlistairMoffat,andJustinZobel.2008.
|                  |     |          |         |     |       |          | Statisticalpowerinretrievalexperimentation. |     |     |     |     |     | InPro- |
| ---------------- | --- | -------- | ------- | --- | ----- | -------- | ------------------------------------------- | --- | --- | --- | --- | --- | ------ |
| cient foundation |     | language | models. |     | arXiv | preprint |                                             |     |     |     |     |     |        |
ceedingsofthe17thACMconferenceonInformation
arXiv:2302.13971.
andknowledgemanagement,pages571–580.
| Hugo Touvron, | Louis | Martin, |     | Kevin | R. Stone, | Peter |     |     |     |     |     |     |     |
| ------------- | ----- | ------- | --- | ----- | --------- | ----- | --- | --- | --- | --- | --- | --- | --- |
Albert, Amjad Almahairi, Yasmine Babaei, Niko- JasonWei, MaartenBosma, VincentY.Zhao, Kelvin
lay Bashlykov, Soumya Batra, Prajjwal Bhargava, Guu, Adams Wei Yu, Brian Lester, Nan Du, An-
|     |     |     |     |     |     |     | drew M. | Dai, | and Quoc | V.  | Le. 2022. | Finetuned |     |
| --- | --- | --- | --- | --- | --- | --- | ------- | ---- | -------- | --- | --------- | --------- | --- |
ShrutiBhosale,DanielM.Bikel,LukasBlecher,Cris-
|                   |     |           |     |                  |     |     | language | models | are | zero-shot | learners. |     | Preprint, |
| ----------------- | --- | --------- | --- | ---------------- | --- | --- | -------- | ------ | --- | --------- | --------- | --- | --------- |
| tianCantónFerrer, |     | MoyaChen, |     | GuillemCucurull, |     |     |          |        |     |           |           |     |           |
arXiv:2109.01652.
DavidEsiobu,JudeFernandes,JeremyFu,Wenyin
| Fu, Brian | Fuller, | Cynthia | Gao, | Vedanuj | Goswami, |     |     |     |     |     |     |     |     |
| --------- | ------- | ------- | ---- | ------- | -------- | --- | --- | --- | --- | --- | --- | --- | --- |
NamanGoyal, AnthonyS.Hartshorn, SagharHos- OrionWeller,DawnJLawrie,andBenjaminVanDurme.
2024. Nevir:Negationinneuralinformationretrieval.
seini,RuiHou,HakanInan,MarcinKardas,Viktor
ConferenceoftheEuropeanChapteroftheAssocia-
Kerkez,MadianKhabsa,IsabelM.Kloumann,A.V.
tionforComputationalLinguistics.
Korenev,PunitSinghKoura,Marie-AnneLachaux,
ThibautLavril,JenyaLee,DianaLiskovich,Yinghai
Lu,YuningMao,XavierMartinet,TodorMihaylov, OrionWeller,KyleLo,DavidWadden,DawnLawrie,
BenjaminVanDurme,ArmanCohan,andLucaSol-
PushkarMishra,IgorMolybog,YixinNie,Andrew
|     |     |     |     |     |     |     | daini. | 2023. | When | do generative |     | query | and docu- |
| --- | --- | --- | --- | --- | --- | --- | ------ | ----- | ---- | ------------- | --- | ----- | --------- |
Poulton,JeremyReizenstein,RashiRungta,Kalyan
|         |                |     |      |        |      |         | mentexpansionsfail? |     |     | acomprehensivestudyacross |     |     |     |
| ------- | -------------- | --- | ---- | ------ | ---- | ------- | ------------------- | --- | --- | ------------------------- | --- | --- | --- |
| Saladi, | Alan Schelten, |     | Ruan | Silva, | Eric | Michael |                     |     |     |                           |     |     |     |
Smith,R.Subramanian,XiaTan,BinhTang,Ross methods, retrievers, and datasets. arXiv preprint
| Taylor, | Adina | Williams, | Jian | Xiang | Kuan, | Puxin | arXiv:2309.08541. |     |     |     |     |     |     |
| ------- | ----- | --------- | ---- | ----- | ----- | ----- | ----------------- | --- | --- | --- | --- | --- | --- |
Xu,ZhengxuYan,IliyanZarov,YuchenZhang,An-
|     |     |     |     |     |     |     | Shitao Xiao, | Zheng | Liu, | Peitian | Zhang, | and | Niklas |
| --- | --- | --- | --- | --- | --- | --- | ------------ | ----- | ---- | ------- | ------ | --- | ------ |
gelaFan,MelanieKambadur,SharanNarang,Aure-
|     |     |     |     |     |     |     | Muennighoff. |     | 2023. | C-pack: | Packaged |     | resources |
| --- | --- | --- | --- | --- | --- | --- | ------------ | --- | ----- | ------- | -------- | --- | --------- |
lienRodriguez,RobertStojnic,SergeyEdunov,and
ThomasScialom.2023b. Llama2: Openfoundation to advance general chinese embedding. Preprint,
| andfine-tunedchatmodels. |     |     | ArXiv,abs/2307.09288. |     |     |     | arXiv:2309.07597. |     |     |     |     |     |     |
| ------------------------ | --- | --- | --------------------- | --- | --- | --- | ----------------- | --- | --- | --- | --- | --- | --- |
EllenMVoorhees.2005. Thetrecrobustretrievaltrack. RuiYang,LinSong,YanweiLi,SijieZhao,YixiaoGe,
|        |       |        |        |     |       |        | XiuLi,andYingShan.2023. |     |     |     | Gpt4tools: |     | Teaching |
| ------ | ----- | ------ | ------ | --- | ----- | ------ | ----------------------- | --- | --- | --- | ---------- | --- | -------- |
| In ACM | SIGIR | Forum, | volume | 39, | pages | 11–20. |                         |     |     |     |            |     |          |
ACMNewYork,NY,USA. largelanguagemodeltousetoolsviaself-instruction.
InAdvancesinNeuralInformationProcessingSys-
Liang Wang, Nan Yang, Xiaolong Huang, Binxing tems,volume36,pages71995–72007.CurranAsso-
| Jiao,LinjunYang,DaxinJiang,RanganMajumder, |     |     |     |     |     |     | ciates,Inc. |     |     |     |     |     |     |
| ------------------------------------------ | --- | --- | --- | --- | --- | --- | ----------- | --- | --- | --- | --- | --- | --- |
11937

ZhiyuanZeng,JiatongYu,TianyuGao,YuMeng,Tanya handlethem-whywouldausergivearetrievalsys-
| Goyal, | and Danqi | Chen. | 2023. | Evaluating | large |                                    |     |     |     |          |     |
| ------ | --------- | ----- | ----- | ---------- | ----- | ---------------------------------- | --- | --- | --- | -------- | --- |
|        |           |       |       |            |       | temaninstructionifitwouldn’tuseit? |     |     |     | However, |     |
languagemodelsatevaluatinginstructionfollowing.
asshownbyTRECwecanseethatthereareuse-
Preprint,arXiv:2310.07641.
casesthatpeoplewanttobeabletodowiththese
Ming Zhao, Peter Anderson, Vihan Jain, Su Wang, instructions. Ourdatasetprovidesthefirststepinto
AlexanderKu,JasonBaldridge,andEugeneIe.2021. buildingsystemsthatcanhandlethembyproviding
Ontheevaluationofvision-and-languagenavigation
|               |                                |     |     |     |     | datasetsfortrainingandtesting. |                 |      | However,systems |                |          |
| ------------- | ------------------------------ | --- | --- | --- | --- | ------------------------------ | --------------- | ---- | --------------- | -------------- | -------- |
| instructions. | arXivpreprintarXiv:2101.10504. |     |     |     |     |                                |                 |      |                 |                |          |
|               |                                |     |     |     |     | should be                      | able to handle  |      | both the        | no-instruction |          |
|               |                                |     |     |     |     | case and                       | the instruction | case | – as            | lots of        | datasets |
A FAQ
existforevaluatingwithno-instructions,ourspro-
videswaystotestfortheinstructioncase.
| Why is     | the dataset | so   | small         | / only around | 100        |     |     |     |     |     |     |
| ---------- | ----------- | ---- | ------------- | ------------- | ---------- | --- | --- | --- | --- | --- | --- |
| instances? | In          | NLP, | 100 instances |               | would be a |     |     |     |     |     |     |
B Llama-3-8BversionofFollowIR
| small dataset | (although |     | not | unheard | of, see Hu- |     |     |     |     |     |     |
| ------------- | --------- | --- | --- | ------- | ----------- | --- | --- | --- | --- | --- | --- |
manEval(Chenetal.,2021),VibeEval(Padlewski WealsotrainaLlama-3-8B(AI@Meta,2024)ver-
et al., 2024), etc. which are commonly used for sionofFollowIR.However,asfoundbymanyoth-
evaluatingLLMsandarewell-respected). ersinthecommunity,Llama3isworsethanMistral
However, in IR, as each query requires anno- forretrieval(BehnamGhaderetal.,2024). Wefind
thatthebasemodelhas0.03p-MRRandafterfine-
| tating hundreds |     | of documents |     | for relevance, | the |     |     |     |     |     |     |
| --------------- | --- | ------------ | --- | -------------- | --- | --- | --- | --- | --- | --- | --- |
numberofqueriesissmallerbutthenumberofan- tuningonFollowIR-trainitgoesto5.1p-MRR.For
notations is similar. Thus, for our 102 query set, nDCG,FollowIR-Llama3-8Bscores15.4whereas
there are roughly 102 queries * (50 relevant and FollowIR-Mistralhas24.8.
150non-relevant)documentsforatotalofaround
C HyperparametersforFine-Tuning
20kannotationstotal–whichissimilartothosein
Mistral
| NLPdatasets. | Thuswecanseethatannotationsfor |          |     |          |            |     |     |     |     |     |     |
| ------------ | ------------------------------ | -------- | --- | -------- | ---------- | --- | --- | --- | --- | --- | --- |
| IR require   | 200x                           | the cost | of  | standard | NLP bench- |     |     |     |     |     |     |
Weusedthefollowinghyperparametersfortrain-
marksperinstance.
|     |     |     |     |     |     | ing Mistral: | batch | size of | 32, cosine | scheduler, |     |
| --- | --- | --- | --- | --- | --- | ------------ | ----- | ------- | ---------- | ---------- | --- |
Thealternativeistogatherlessannotationsper 3e-5 learning rate, max length of 2048, lora rank
query(suchasNQorMSMarcowhichonlyhave 8andalpha16,bfloat16,andtrainedfor8epochs.
| one relevant | document |     | per query | but | many more |         |                               |     |     |     |         |
| ------------ | -------- | --- | --------- | --- | --------- | ------- | ----------------------------- | --- | --- | --- | ------- |
|              |          |     |           |     |           | We used | "q_proj,v_proj,o_proj,k_proj" |     |     |     | for the |
queries)butthosehavebeenshowntooverwhelm-
|                                            |              |        |     |          |             | LoRA tuning       | and trained |     | from mistralai/Mistral- |     |     |
| ------------------------------------------ | ------------ | ------ | --- | -------- | ----------- | ----------------- | ----------- | --- | ----------------------- | --- | --- |
| inglycontaindocumentsthatarerelevantbutnot |              |        |     |          |             | 7B-Instruct-v0.2. |             |     |                         |     |     |
| marked                                     | as relevant, | making |     | them low | quality for |                   |             |     |                         |     |     |
evaluation (Ni et al., 2021; Cai et al., 2022). As D HyperparametersforInference
suchtheIRcommunitydevelopsmorethoroughly
Weusedefaultparametersforinference,takenfrom
judgedversionsofthemwithasmallernumberof
|         |          |          |          |        |       | the original | code of | the authors | of  | the papers | we  |
| ------- | -------- | -------- | -------- | ------ | ----- | ------------ | ------- | ----------- | --- | ---------- | --- |
| queries | (such as | the Deep | Learning | Tracks | 2019- |              |         |             |     |            |     |
use(fromtheirMTEBevaluations).
| 2022 that               | build | off of | MSMarco, | cited | above, or |               |     |     |     |     |     |
| ----------------------- | ----- | ------ | -------- | ----- | --------- | ------------- | --- | --- | --- | --- | --- |
| thosethatwebuildoffof). |       |        |          |       |           | E ComputeUsed |     |     |     |     |     |
CanLMshandlethesetypeoflonginstructions? WeusedaA10080GBfortheexperiments. Train-
Yes! Asthelengthofthequeryplusinstructionis
ingtookroughly6hourswhileinferencetookbe-
roughly 80 words on average and the documents tween3-12hoursaccordingtomodelsize.
| are around | 400 | words, | most | (if not all) | LMs can |     |     |     |     |     |     |
| ---------- | --- | ------ | ---- | ------------ | ------- | --- | --- | --- | --- | --- | --- |
handle 480 words in their context length. This F FullRetrievalResults
especiallyholdstrueformodernLMswith>1024
|               |          |     |          |         |            | In Table   | 5 we show        | results | for         | models | search-   |
| ------------- | -------- | --- | -------- | ------- | ---------- | ---------- | ---------------- | ------- | ----------- | ------ | --------- |
| token context | lengths. |     | Overall, | this is | really not |            |                  |         |             |        |           |
|               |          |     |          |         |            | ing on the | full collections |         | of the TREC |        | tasks in- |
longcontextforLMs.
|     |     |     |     |     |     | cluded | in FOLLOWIR. | Note | that | because | each |
| --- | --- | --- | --- | --- | --- | ------ | ------------ | ---- | ---- | ------- | ---- |
Not all retrieval tasks will have instructions, modelretrievesdifferentrelevantdocuments,the
whatcanwedothen? Weagreethatthisisthe instruction-followingevaluationhasadifferentset
case, in fact, we could only find this one source ofinstancesthateachmodelisevaluatedon(asit
ofreal-worldinstructionsforretrieval! Webelieve can only be evaluated on documents it retrieved
thisispartiallyduetothelackofsystemsthatcan thatthenbecomenot-relevant).
11938

|     |     |     |       | Robust04  | News21    | Core17    |
| --- | --- | --- | ----- | --------- | --------- | --------- |
|     |     |     |       | (mAP)     | (nDCG@5)  | (mAP)     |
|     |     |     | Model | OG        | ∆ OG      | ∆ OG ∆    |
|     |     |     | BM25  | 21.4 -1.2 | 30.1 +5.3 | 16.8 -0.2 |
tcurtsnI
|     |     | oN  | E5-base-v2        | 22.7 -7.0  | 33.6 +1.8 | 19.7 -3.0 |
| --- | --- | --- | ----------------- | ---------- | --------- | --------- |
|     |     |     | Contriever        | 19.2 -7.7  | 22.5 +9.0 | 22.6 -7.6 |
|     |     |     | TART-Contriever   | 25.5 -10.1 | 40.0 -5.0 | 22.6 -7.6 |
|     |     |     | tcurtsnI BGE-base | 23.6 -3.1  | 36.5 -7.8 | 23.0 -2.1 |
sesU
|     |     |     | INSTRUCTOR-base | 22.5 -2.2 | 33.3 -2.8 | 20.0 -0.2 |
| --- | --- | --- | --------------- | --------- | --------- | --------- |
|     |     |     | INSTRUCTOR-XL   | 30.4 -3.1 | 38.1 -0.1 | 29.9 -2.8 |
Table5: FOLLOWIR scoresonthefullretrievalcollection(thusrerankersarenotincluded). Asthebasescore
isdifferent,therearedifferentnumbersofrelevantdocumentstheyarebeingevaluatedonforp-MRR.Thus,we
onlyreporttheoriginal(no-instruction)scoreandthedeltawhenusingtheTRECinstructions. Wenotethatit
showssimilarresultstothemaintext–retrievalmodelsarenoteffectivelyusinginstructionsandseeperformance
degradationswithlongertext.
G KeywordsusedforBEIRexperiments
| GPT-4-Turbo-1106 |     | extracted | the following | key- |     |     |
| ---------------- | --- | --------- | ------------- | ---- | --- | --- |
words(Table6)fromtheinstructionsthesemodels
used,whichgeneratedtheresultsinTable3.
H PromptsUsed
Weusethesepromptsforgeneratingtheshortin-
| structions, | the keywords, |     | and the synthetic | doc-       |     |     |
| ----------- | ------------- | --- | ----------------- | ---------- | --- | --- |
| uments.     | The examples  |     | used in the       | prompt for |     |     |
the“FullInstructionstoShortInstructions"prompt
| were partially | created |     | by the authors, | as only |     |     |
| -------------- | ------- | --- | --------------- | ------- | --- | --- |
theshortinstructionswereprovidedbyTART/IN-
STRUCTOR.
11939

| Model                     | Dataset  |                               |                       | Keywords     |
| ------------------------- | -------- | ----------------------------- | --------------------- | ------------ |
| BM25/Contriever/E5/MonoT5 | FiQA     |                               |                       | Financeweb   |
| BM25/Contriever/E5/MonoT5 | SciFact  |                               | sciencepaperverify    |              |
| BM25/Contriever/E5/MonoT5 | NFCorpus |                               | medicinerelevant      |              |
| TART-dual                 | FiQA     |                               |                       | financialweb |
| TART-dual                 | SciFact  |                               | scientificpaperverify |              |
| TART-dual                 | NFCorpus | scientificpaperparagraph      |                       |              |
| INSTRUCTOR-base           | FiQA     |                               | financialsupporting:  |              |
| INSTRUCTOR-base           | SciFact  | scientificsupportingpassage:  |                       |              |
| INSTRUCTOR-base           | NFCorpus |                               | medicinerelevant      |              |
| BGE-base                  | FiQA     |                               | relevantpassages:     |              |
| BGE-base                  | SciFact  |                               | relevantpassages:     |              |
| BGE-base                  | NFCorpus |                               | relevantpassages:     |              |
| INSTRUCTOR-xl             | FiQA     |                               | financesupporting:    |              |
| INSTRUCTOR-xl             | SciFact  | scientificsupportingpassages: |                       |              |
| INSTRUCTOR-xl             | NFCorpus | nutritionfactspublicmedical:  |                       |              |
| E5-Mistral                | FiQA     |                               | financialreplies      |              |
| E5-Mistral                | SciFact  |                               |                       | scientific   |
| E5-Mistral                | NFCorpus |                               | retrieverelevant      |              |
| TART-T5-FLAN-xl           | FiQA     |                               |                       | financialweb |
| TART-T5-FLAN-xl           | SciFact  |                               | scientificpaperverify |              |
| TART-T5-FLAN-xl           | NFCorpus | Scientificpaperparagraph      |                       |              |
Table6: KeywordsusedfortheBEIRkeywordanalysis. Notethatnon-instructionmodelsreceivedthekeywords
usedinINSTRUCTOR-baseandTART-dual(asshowninthetable).
11940

SyntheticDocumentCreation
I need you to annotate some data for my business and it is super important that you follow
instructionspreciselyoryouwillbefired.
GivenaGooglesearchqueryandinstructionsregardingwhatmakesadocumentrelevant,Ineed
youtowritetwodocuments: onethatwouldberelevantandonethatwouldnot.
Search: TITLE_HERE
Instructions: NARRATIVE_HERE
I need some different options to choose from, so give me three **different** options for both
a relevant document and an irrelevant document. They should be **long** paragraph-sized
documents( 300wordseach),oneoneachline. Ifthereisnonegationintheinstructions,your
∼
irrelevantdocumentshouldbeslightlyofftopic:
ShortInstructionstoKeywords
Ihaveinstructionsthatarespecifictoastyleofretrieval,butIwantyoutoinsteadjustfocusonthe
relevantkeywordsthatareintheseinstructions. Yourjobistoreturnalistofthesekeywordsthat
arerelevantinthequery. Thereareprobablyoneortworelevantkeywordstoextractonly.
##Examples
###Example1:
Instruction: HelpmetofindahighlyrelatedPubMedpapertoanswerthisquestion.
Keywords: ["PubMed"]
###Example2:
Instruction: IwanttofindananswerforthisTriviaquestion. Canyoufindsomeparagraphsthat
provideevidencefromWikipedia?
Keywords: ["Trivia","Wikipedia"]
###Example3:
Instruction: CheckifaQuoraquestionisduplicatedwiththisquestion.
Keywords: ["Quora","duplicated"]
###Example4:
Instruction: IwanttofindarelatedquestionaskedinStackExchange. Canyoufindoneforme?
Keywords: ["related","StackExchange"]
##Yourturn
Instruction: FILL_TEXT_HERE
Keywords (either one or two keywords, that are not "documents", "questions", "answer", or
"articles"):
11941

FullInstructionstoShortInstructions
Ihaveinstructionsthatarespecifictoaquestion,butIneedyourhelpabstractingthemtoageneraltaskformatthatI
cangivetosomeoneelse.Ineedyoutoturnthemintoanabstractcommandthatjustdescribethegeneralabstracttask
instead(e.g.,wherethedataisfrom,whatthetypeofdocumentlookslike).Itiscrucialthatyoureadandfollowthese
instructions,youwillgetalargebonusifyouaresuccessful($200).
Theabstractcommandshouldonlymentionthe**taskformat**.Do**not**refertoanyentitiesorspecifictextinthe
originalinstruction.Yourresponseshouldbearound10words.Thecommandshouldbeasifyouwerespeakingto
anotherhuman.
##Examples
###Example1:
OriginalInstruction:Arelevantdocumentwouldprovideinformationaboutthewholeblood-baseperfusateandwhether
ornot itprovidessuperiorpreservationof myocardialfunction duringex vivoheart perfusion. Thismayinclude
researchexperiments,commentary,orsurvey/reviewpapers.Informationaboutwholeblood-baseperfusatealoneisnot
relevant,unlessitalsomentionsit’seffectonmyocardialfunctionduringexvivoheartperfusion.
AbstractCommand:HelpmetofindahighlyrelatedPubMedpapertoanswerthisquestion.
###Example2:
OriginalInstruction:Arelevantdocumentwillcontaininformationthatabouttherightofwayininternationalwaters
thatcanbeusedtodeterminewhoshouldhavetherightofwayinagivensituation. Forexample,itshouldcontain
instancesaboutwhoisatfaultinanaccident,ifitdependsonthesizeoftheboat,ordetailsabouthowthisdiffers
accordingtonationality.Especiallyrelevantaredocumentsdescribingwhoisatfaultinacrashsituation.
AbstractCommand:RetrieveaWikipediaparagraphthatanswersthisquestion.
###Example3:
OriginalInstruction: Arelevantinstancewillbeaquestionthatissemanticallyequivalenttothequerygiven. For
example,itmaycontaindifferentlexicalwordsorbeaparaphraseoftheother,buttheunderlyingmeaningwillbethe
same.Iftheinstanceisnotsemanticallythesameasthequery,itisirrelevant.
AbstractCommand:CheckifaQuoraquestionisduplicatedwiththisquestion.
###Example4:
OriginalInstruction:Arelevantdocumentwouldincludedetailsaboutthetimingofmedicareandwhatagepatientscan
startusingitforhealthcare.Itmayincludeinformationaboutlaws,insurance,orotherdetailsthatdescribetheagethe
medicarebegins.Lessrelevantaredocumentstalkingaboutpotentiallawsorfactsaboutmedicarethatdonotanswer
thequestionofwhatagemedicarebegins.Justthementionofanageandwhentostartmedicarewouldberelevant.
AbstractCommand:Iwanttoknowtheanswertothequestion.Canyoufindgoodevidenceontheweb?
Nowyoucaneasilyseethattheabstractcommandisvagueanddescribesonlyashortcommandabouthowtogetthe
informationyouneed. Followthisexactly—donotreferencespecifics(likeintheabove,"internationalwaters"and
"medicare"arenotincludedintheabstractcommand).Youshouldinsteadkeeptheabstractcommandvagueandwell,
abstractaboutthetaskonly.Usetheword"question".
##Yourturn
OriginalInstruction:FILL_TEXT_HERE
AbstractCommand(remembertousetheword"question"):
11942
