PublishedasaconferencepaperatICLR2025
PROMPTRIEVER: INSTRUCTION-TRAINED RETRIEVERS
CAN BE PROMPTED LIKE LANGUAGE MODELS
OrionWeller∗ι BenjaminVanDurmeι DawnLawrieι
AshwinParanjapeα YuhaoZhangα JackHesselα
ιJohnsHopkinsUniversity αSamayaAI
oweller@cs.jhu.edu
ABSTRACT
Instruction-tunedlanguagemodels(LM)areabletorespondtoimperativecom-
mands,providingamorenaturaluserinterfacecomparedtotheirbasecounterparts.
Inthiswork,wepresentPromptriever,thefirstretrievalmodelabletobeprompted
like an LM. To train Promptriever, we curate and release a new instance-level
instructiontrainingsetfromMSMARCO(Nguyenetal.,2016),spanningnearly
500kinstances. Promptrievernotonlyachievesstrongperformanceonstandard
retrievaltasks,butalsofollowsinstructions. Weobserve: (1)largegains(reaching
SoTA)onfollowingdetailedrelevanceinstructions(+14.3p-MRR/+3.1nDCG
onFollowIR),(2)significantlyincreasedrobustnesstolexicalchoices/phrasingin
thequery+instruction(+12.9Robustness@10onInstructIR),and(3)theabilityto
performhyperparametersearchviapromptingtoreliablyimproveretrievalperfor-
mance(+1.4averageincreaseonBEIR).Promptrieverdemonstratesthatretrieval
modelscanbecontrolledwithpromptsonaper-querybasis,settingthestagefor
futureworkaligningLMpromptingtechniqueswithinformationretrieval. 1
1 INTRODUCTION
Moderninformationretrieval(IR)modelsgenerallymatchqueriestopassagesbasedonasingle
semanticsimilarityscore.Asaresult,theuserexperienceofsearchcanbeopaque,withusersneeding
tofindparticularkeywords/phrasings,applyvariousfiltersinadvancedsearchsettings,anditerate
basedonprevioussearchestofindthe“justright"querythatreturnstherightpassages.
In this work, we introduce Promptriever: a retrieval model that can instead be controlled via
naturallanguageprompts. Forexample, ifauserissearchingforJamesCameronmovies, butis
only interested in movies prior to 2022 that are not co-directed, instead of applying a series of
searches/hardfilters,Promptrievercanadjustitsnotionofrelevancedynamicallybasedonanatural
languagedescription: “Relevantdocumentsarenotco-directed,andarecreatedbefore2022"(see
Figure1foracomparison).
Promptrieverisabi-encoderretriever;itsbackboneisalargelanguagemodel(LM)suchasLLaMA-2
7B(Touvronetal.,2023a;b).2 BeforeIRtraining,languagemodelscanreadilyadapttheiroutputs
basedonnaturallanguage(alsoreferredtoasinstructionsorprompts). ButafterstandardIRtraining,
whichtypicallyfocusesonoptimizingasinglequery-passage“semanticsimilarity"score,instruction
followingcapacityisnotmaintained(see§5). Whilesomerecentworkhasaddedinstructionsto
retrievalmodeltrainingsets(Suetal.,2022;Asaietal.,2022;Jiangetal.,2023;Muennighoffetal.,
2024)thesearetemplated“instructions",prependedtoeveryqueryinthedatasetattrainingandtest
time,ratherthanlanguagedefiningrelevanceconditionsonaper-instancebasis. Forexample,when
usingtheMSMARCOdataset,thestandard“instruction"is“Givenawebsearchquery,retrieve
∗WorkperformedduringaninternshipatSamayaAI.
1Codeanddataareavailableathttps://github.com/orionw/promptriever
2Weexperimentwithmultiplebackbonesin§5.1.
1

PublishedasaconferencepaperatICLR2025
relevantpassagesthatanswerthequery"whichisprependedtoeveryquery(Muennighoffetal.,
2024;Wangetal.,2023).
| Promptriever, | in  | contrast, | is trained | to main- |     |     |     |     |     |
| ------------- | --- | --------- | ---------- | -------- | --- | --- | --- | --- | --- |
tain the per-instance instruction following ca- Query Capabilities
| pabilityofitsbackboneLM.Todoso, |     |     |     | wecu- |     |     |     |     |     |
| ------------------------------- | --- | --- | --- | ----- | --- | --- | --- | --- | --- |
~
| rate and | release | a synthetic | dataset | of 500K | Standard |     |     |     |     |
| -------- | ------- | ----------- | ------- | ------- | -------- | --- | --- | --- | --- |
Query: What movies were
query-passagerelevancepairsaugmentedwith Retrieval created by James Cameron?
| instance-levelinstructions. |     |     | Thetwokeynovel- |     |     |     |     |     |     |
| --------------------------- | --- | --- | --------------- | --- | --- | --- | --- | --- | --- |
tiesinthetrainingsetare: (1)instructionsdefin- Current Given a web search query, retrieve
|     |     |     |     |     | Instructable |     | relevant passages: What movies |     | Dataset |
| --- | --- | --- | --- | --- | ------------ | --- | ------------------------------ | --- | ------- |
ingper-queryrelevanceindiverse,free-formnat-
|                                             |     |                               |     |     | Retrievers |                                   | were created by James Cameron? |     | Prefix   |
| ------------------------------------------- | --- | ----------------------------- | --- | --- | ---------- | --------------------------------- | ------------------------------ | --- | -------- |
| urallanguage;                               |     | and(2)instance-level“instruc- |     |     |            |                                   |                                |     |          |
| tionnegatives."Instructionnegativesarecases |     |                               |     |     |            |                                   |                                |     | Query    |
|                                             |     |                               |     |     |            | What movies were created by James |                                |     | specific |
where the (query, passage) pair is highly rele- Cameron? Relevant docs are not co-
|     |     |     |     |     | Promptable |     |     |     | relevance |
| --- | --- | --- | --- | --- | ---------- | --- | --- | --- | --------- |
vantifviewedinisolation,but,theadditionof Retrievers directed and created before 2022. Be definition
careful in your response and be sure to
acarefullyconstructedinstructionsignificantly be correct because I will tip you $1000 Prompt
| decreases          | that relevance. |     | For example,        | a gen- |     |     |     |     |     |
| ------------------ | --------------- | --- | ------------------- | ------ | --- | --- | --- | --- | --- |
| erated instruction |                 | can | request additional, | fine-  |     |     |     |     |     |
grained information not present in a topically Figure 1: An illustration of the capabilities of
relevant passage, as in Figure 2. By construc- retrieval models. Standard retrieval models find
|     |     |     |     |     | semantic | similarity | to the input | query, | typically |
| --- | --- | --- | --- | --- | -------- | ---------- | ------------ | ------ | --------- |
tion,toachievelowtraininglossoninstruction
negatives,modelsmustadapttheirnotionofrel- matchingusingquerykeywordsandphrases. Cur-
evance per-query by the instruction, teaching rentinstructableretrieversprependadatasetprefix
themtobesensitivetofine-graineddetails. thatgenericallydescribesthetask,isusedforall
|     |     |     |     |     | queriesinthedataset, |     | andisalsogenerallyused |     |     |
| --- | --- | --- | --- | --- | -------------------- | --- | ---------------------- | --- | --- |
Promptrievernotonlyachievesstrongretrieval during training. We propose promptable retriev-
| scores in | standard | settings, | but | also follows |     |     |     |     |     |
| --------- | -------- | --------- | --- | ------------ | --- | --- | --- | --- | --- |
erswhichcanhandlecomplexinstructionsinclud-
| instructions | more | effectively | than | prior mod- |              |           |             |     |           |
| ------------ | ---- | ----------- | ---- | ---------- | ------------ | --------- | ----------- | --- | --------- |
|              |      |             |      |            | ing detailed | relevance | definitions | and | zero-shot |
els. Furthermore,weshowtheeffectivenessof
|     |     |     |     |     | prompting | techniques | that act | as a form | of zero- |
| --- | --- | --- | --- | --- | --------- | ---------- | -------- | --------- | -------- |
thesemethodsbycomparingtothesamerecipe
shothyperparameteroptimization,justasprompt-
with and without instructions. We show that ingcanbedonewithlanguagemodels.
instruction-tuningprovides:
• State-of-the-artbi-encoderperformanceoninstruction-followingretrievaltasks(+14.3p-MRR,+3.1
nDCG/MAPonFollowIR(Welleretal.,2024))withcomparablescorestoSoTAcross-encoders.
• Improvedrobustnesstoquerylength/phrasingwitha44%decreaseinvarianceacrossinstructions
onBEIR(Thakuretal.,2021)and+12.9Robustness@10onInstructIR(Ohetal.,2024).
• Reliableimprovementsinretrievalperformancezero-shotsolelybyprompting(suchasadding
“ThinkcarefullywhenassigningrelevanceandIwillgiveyouatip"),enablingpromptengineering.
Our results demonstrate that with the right training data, modern bi-encoders can be instruct-
ed/promptedinfree-formnaturallanguage,inasimilarmannertoLMs. Wehopethisalignment
betweentheLMandIRcommunitieswillallowforfurtherimprovementstodenseretrievalmodels.
| 2 PROMPTRIEVER |     |     | DATA | GENERATION |     |     |     |     |     |
| -------------- | --- | --- | ---- | ---------- | --- | --- | --- | --- | --- |
Totrainabi-encoderthatcanretrievebasedoninstructions,westartwiththepopularIRtraining
dataset,MSMARCO(Nguyenetal.,2016). Weusethetevatron-msmarco-augversionwhich
includeshard-negativesandwasusedtotrainRepLLaMA(Maetal.,2023). Thistrainingsetincludes
roughly491kqueriesandprovidesapositivepassageand30hardnegativesforeach. Weaugment
thissetwithinstructionsinatwopartprocess: (1)instructiongenerationfromtheinitialqueriesand
| (2)instruction-negativemining. |     |     |     | TheseprocessesareillustratedinFigure2. |     |     |     |     |     |
| ------------------------------ | --- | --- | --- | -------------------------------------- | --- | --- | --- | --- | --- |
2.1 INSTRUCTIONGENERATION
We start by generating an instruction3 for each query in MS MARCO. Consider the example in
Figure2wheretheinitialquery(“Whichtypeofvolcanoeruptionhasnotbeenseen?")isseeking
3SeeAppendixDforadefinitionofinstructions.
2

PublishedasaconferencepaperatICLR2025
Query
Instruction Positive Instruction positive
Query positive
Which type of volcano eruption has not been seen?
Subglacial Volcanoes: Unseen Eruptions
Original Positive Subglacial volcanoes, which erupt beneath
ice sheets or glaciers, have not been
What are the differences of maar and caldera? directly observed erupting. These volcanoes
Calderas form .. An explosive caldera-forming are formed when ....
eruption has never been witnessed first hand ..
formation: maars form ... Instruction Negative Instruction negative
Query positive
Instruction
Impact of Volcanic Eruptions on Climate
Volcanoes are classified into different types based on Volcanic eruptions can have significant
their shape and eruption style. A document is relevant impacts on the climate ... e.g. the eruption of
if it describes a specific type of volcano that has Mount Tambora ... there are some types of
not been directly observed erupting, and provides eruptions that have not been witnessed first
information about its formation or characteristics. hand, which could provide further insights...
Figure2: Thedatagenerationprocesstogenerateinstruction-basedretrievaldata. Wetaketheinitial
query and relevant passage and prompt an LM to generate an instruction that would match that
query. Note that the instruction adds extra qualifications to the definition of relevance. We then
askanLMtogenerateanexamplerelevantandnon-relevantpassageforthatqueryandinstruction.
Weseethatthegeneratedpositivepassagefulfillstheextrarequirement(inpink)butthegenerated
instruction-negativedoesnot. Wegeneratemultipletypesofinstructions(bothinlengthandstyle).
just a type of volcano. We ask an LM to generate instructions that add additional requirements,
explicitlyexcludecertaintypesofpassages,oruseambiguitytomaketheinitialquerymorespecific.
For example, in the volcano query, the generated instruction asks for both the volcano type and
informationaboutitsformation(inpink). WeuseLlama-3-70B-Instruct(AI@Meta,2024).4
Instructiondiversity Toensureawiderangeofinstructions,weaskLlama3togenerateinstruc-
tionsof: 1)varyinglengthformats(fromshortone-sentenceinstructionstotwoparagraph-sized5
instructions),and2)differing“styles",whichcaneitherbeapersonaofthepersongivingthequery,
negationofsomeaspect,orgenericbackgroundinformation. Thegeneratedvolcanoinstructionin
Figure2hasthegenericbackground“style"andwasrequestedtobetwosentenceslong.
Overall,Llama3succeededinfollowinglength+stylespecifications(Table6). Aqualitativeexplo-
rationofexamplesisintheAppendix: bystyleinTable8,andbylengthinTable9.
Maintaining original MS MARCO positive relevance We aim to generate instructions that
maintain the positive relevance relation between (query, passage) pairs from MS MARCO. We
provideboththequeryandpositivepassagetotheLMwhengeneratinginstructions,andrequestthat
themorespecificinstructionkeepthepassagerelevant. Tocheckforsuccessinthisregard: weuse
FollowIR-7B(Welleretal.,2024),across-encodercapableofmakingnuancedrelevancejudgments
regarding(query,instruction,passage)instances. FollowIR-7Bmarkedroughly15%ofthegenerated
instructionsasmakingtheoriginalpositivepassagenolongerrelevant. Inthesecases,wesubstitute
theoriginalpositivedocumentwithonegeneratedfromthenextstage.6
2.2 INSTRUCTIONNEGATIVEMINING
Aftergeneratingtheinstructions,it’spossibletotrainmodelsusingtheexactsamedataasRepLLaMA,
exceptthequerieshavebeenaugmentedwithinstructions.However,ourinstructed-augmentedqueries
havenotchangedanyrelevancerelationsinthecorpus: withnoadditionalmodification, models
couldsimplyignoreourinstructionsentirely,andachievethesameperformance.7
Thus,wedevelopacomplementarydataaugmentationthatencouragesmodelstopayattentiontothe
instruction. Wetermthisaugmentationinstructionnegatives:8 wherethepassageisquery-positive
4SeeAppendixFforpromptdetails
5Whilewedon’texpectrealusersofIRsystemstoregularlytypetwo-paragraphinstructions,includingthese
helpsthemodellearndiversityoflengthandadaptto(potentiallymachinegenerated)diverserequests.
6Weevaluated20oftheseinstructionsthatpassedthefilterandfoundallpositivepassageswerestillrelevant.
7AsweshowasanablationinTable4,inthissetting,modelsindeeddoignoretheinstruction.
8ThisissimilartotheintuitionofAsaietal.(2022)butcruciallytheinstruction-negativesaregatheredatan
instancelevelratherthanatthedataset-level.
3

PublishedasaconferencepaperatICLR2025
butinstruction-negative,i.e.,whentheinstructionisaddeditdecreasesthepassage’srelevance. To
achievelowtrainingloss,themodelmustlearntoconditiononbothqueryandinstruction.
Initial attempts to gather instruction negatives from the MS MARCO collection itself fell short:
qualitatively,wefoundthatthecorpusdoesnotcontainenoughpositivepassagesperquerytomine
nuancedquery-positivesbutinstruction-negatives. Thus,weturntogeneratingpassageswithaLM.
Weusegpt-4o-2024-05-13togeneratetheinstructionnegativepassages,generatingonequery-
positive/instruction-positivepassageandthreequery-positive/instruction-negativepassagesper(query,
instruction) pair. We over-generate candidates and then filter them post-hoc because initial test-
ing revealed that (on average) only two out of three generated passages were correctly query-
positive/instruction-negative.9 We again use FollowIR-7B for the filter by: 1) checking that the
generated instruction negatives are actually instruction-negative (and discarding it if not) and 2)
checkingthatthegeneratedinstruction-positivewasactuallyrelevant(anddiscardingifnot).
Filtering validation Wetasked4 human annotators withthe filtration task: for a given (query,
instruction,generatedpassage)triplet,isthepassagerelevantornottothequery+instruction. For
thistask,averagehuman-humanagreementwas75%(N=32),whereas,theaveragehuman-model
agreementwas84%(N=64). Thus,FollowIR-7Bactsasasufficientlyhigh-qualityfilter.
3 EXPERIMENTAL SETTINGS
Ourgoalistoshowtheefficacyofinstruction-traininginIRcomparedtostandardtraining. Thus,we
primarilycomparetoRepLLaMAandusetheirdata/recipeforapples-to-applescomparison.
3.1 TRAINING
We train Promptriever on the RepLLaMA MS MARCO data as well as the new instruction data
generatedbyLlama3andGPT-4o. Weusethesamelearningrateandotherhyperparameterdetailsas
theoriginalRepLLaMAforafaircomparison(seeAppendixEformoredetails).10 Weuseallvalid
instruction-negativesintrainingandsampletheremainderofthehard-negativesfromthedatasetused
totrainRepLLaMA(keepingthesamenumberofhardnegativesperquery).
3.2 EVALUATIONDATASETS
Weevaluateonin-domain(MSMARCO),out-of-domain(Thakuretal.,2021,BEIR),andinstruction-
followingretrievaldatasets,includingInstructIR(Ohetal.,2024)andFollowIR(Welleretal.,2024).
Notethattheseinstruction-followingdatasetsevaluateinstructionsonaper-querybasis.
ThemetricsusedarenDCG@10forBEIR,TRECDL19(Craswelletal.,2020)andDL20(Soboroff,
2021),andMRRforMSMARCODev.
Theinstruction-followingdatasetsusebothstandardandinstruction-followingmetrics: nDCG@5for
theNews21portionofFollowIR,MAP@1000fortheCore17andRobust04portions,aswellasusing
p-MRRforallportions(rangingfrom-100to100)whichmeasuresthesensitivitytoinstructions
intheprompt(-100meansitfollowstheoppositeoftheinstruction,0meansnochange,and100
meansperfectinstructionfollowing). InstructIRusesnDCG@10aswellasRobustness@10which
measurestheminimumnDCG@10scoreover10differentprompts. Higherisbetterforallmetrics.
WeprimarilycomparewithRepLLaMAtohaveanapples-to-applescomparison. However,wealso
showresultsformanyothermodels,includingMonoT5(Nogueiraetal.,2020),Instructormodels
(Suetal.,2022),thebi-encoderTARTmodeltrainedfromContriever(Asaietal.,2022),E5Mistral
(Wangetal.,2023),GoogleGecko(Leeetal.,2024),andBM25(Robertsonetal.,1995;Lù,2024).
9WefoundthatcurrentLMsstrugglewiththisnuanceandthatonlythemostcapableLMs(e.g.GPT-4obut
notGPT-4o-mini)wereabletoproduceinstructionnegativeswithhighenoughefficiencytobepractical.As
describedintheprevioussection,weusethequery-positive/instruction-positivepassageasthebackuppositive
passageiftheoriginalpositivepassagewasnotrelevanttothegeneratedinstruction.
10NotethatalthoughthismeansthatPromptrieverhasseenmoredataoverall,weprovidecomparisonsin
Section5controllingfortrainingdatavolume.Ourfindingsdonotchange(i.e.,thatPromptriever’sinstruction
followingcapacityisnotsimplyaresultofseeingmoredatapoints).
4

PublishedasaconferencepaperatICLR2025
Table1: ResultsforinstructionfollowingontheFollowIRandInstructIRdatasets. Higherisbetter
forallmetrics;MAP@1000/NDCG@5/Robustness@10rangefrom0-100;p-MRRrangesfrom-100
to100. Despiteusingthesamebackbonemodel(Llama2)andrecipe,Promptrieversignificantly
outperformsRepLLaMAwitha+3.1gainonthestandardretrievalscore(nDCG/MAPaverage)and
a+14.3pointgainonp-MRR.Promptrieveroutperformsallotherdenseretrievermodelsand
scorescomparablytothebestcross-encoder,FollowIR-7B,despitenotusingattentionbetween
queryanddocuments. GeckoscoresuseaproprietaryAPIandwerenotreportedforInstructIR.Bold
resultsindicatethebestforthatarchitecturetype(e.g. cross-encoder,bi-encoder)whileasterisksare
statisticalsimilaritytothebest(forbi-encodersonly,ascross-encodersarenotcomparable§B).
|           |     |     |          |       | FollowIR   |        |       |             |      | InstructIR   |
| --------- | --- | --- | -------- | ----- | ---------- | ------ | ----- | ----------- | ---- | ------------ |
| Model     |     |     | Robust04 |       | News21     | Core17 |       | Average     |      | MSMARCO      |
|           |     |     | MAP      | p-MRR | nDCG p-MRR | MAP    | p-MRR | Score p-MRR |      | nDCG Robust. |
| MonoT5-3B |     |     | 27.3     | +4.0  | 16.5 +1.8  | 18.2   | +1.8  | 20.7        | +2.5 | - -          |
sredocnE
ssorC Mistral-7B-instruct 23.2 +12.6 27.2 +4.8 19.7 +13.0 23.4 +10.1 63.1 35.3
FollowIR-7B 24.8 +13.7 29.6 +6.3 20.0 +16.5 24.8 +12.2 81.3 71.5
| BM25 |     |     | 12.1 | -3.1 | 19.3 -2.1 | 8.1 | -1.1 | 13.2 | -2.1 | 76.0 26.9 |
| ---- | --- | --- | ---- | ---- | --------- | --- | ---- | ---- | ---- | --------- |
TART-Contriever 14.3 -9.0 21.8 -3.0 13.3 -3.0 16.5 -5.0 84.8 47.5
sredocnEiB
InstructorXL 19.7 -8.1 26.1 -0.9 16.8 0.7 20.9 -2.8 48.6 21.5
| E5-Mistral  |     |     | 23.1 | -9.6 | 27.8 -0.9   | 18.3  | +0.1 | 23.1  | -3.5 | 86.3 55.4 |
| ----------- | --- | --- | ---- | ---- | ----------- | ----- | ---- | ----- | ---- | --------- |
| GoogleGecko |     |     | 23.3 | -2.4 | *29.5 *+3.9 | *23.2 | +5.4 | *25.3 | +2.3 | - -       |
| RepLLaMA    |     |     | 24.0 | -8.9 | 24.5 -1.8   | 20.6  | +1.3 | 23.0  | -3.1 | 85.7 50.2 |
Promptriever *28.3 *+11.7 *28.5 *+6.4 *21.6 *+15.4 *26.1 *+11.2 *92.1 *63.1
4 RESULTS
PromptrieveroutperformstheoriginalRepLLaMAininstructionfollowing(§4.1)whilemaintaining
strongstandardretrievalperformance(§4.2). WealsodemonstratethatPromptrievercanbereliably
zero-shotprompted,inthesamemannerasalanguagemodel(§4.3).
Wefurthershowthatthesearestatisticallysignificant,andcomputeatwo-sidedstudentt-test(p<
0.05)fornDCGandMAPmetricsandWilcoxonsignedranknon-parametrictest(duetoanon-normal
distribution)forp-MRRandRobustness@10(alsousingp<0.05)
4.1 INSTRUCTIONFOLLOWING
Table1presentstheresultsfortheFollowIRandInstructIRdatasets. Promptrieveristhehighest
performingdenseretriever,improvingoverRepLLaMAby+14.3p-MRR(-3.1→+11.2)and+3.1in
nDCG/MAP.Forreference,wealsoincludetheresultsfromthreecomputationallyintensivecross-
encodermodels. Whilecross-encoders(asexpected)performbestduetotheirsignificantcompute
advantage,Promptrieverachievescomparablescoresasamuchmoreefficientbi-encodermodel. Our
model’sstrongperformanceversustheRepLLaMAbaselineillustratesthatourinstructiondatais
highlyeffective,leadingtosignificantgainsininstructionfollowingandpromptrobustness.11
4.2 IN-DOMAINRETRIEVAL
|     |     |     |     |     | Table 2: | MS MARCO    | (in-domain). |              | We  | see that |
| --- | --- | --- | --- | --- | -------- | ----------- | ------------ | ------------ | --- | -------- |
|     |     |     |     |     | they are | comparable, | despite      | Promptriever |     | being    |
WebenchmarkPromptrieveronthree
|          |           |       |      |     | instruction-trained. |     | Boldindicatesbestinthecol- |     |     |     |
| -------- | --------- | ----- | ---- | --- | -------------------- | --- | -------------------------- | --- | --- | --- |
| standard | retrieval | tasks | from | in- |                      |     |                            |     |     |     |
umn,asterisksarestatisticalsimilaritytothebest.
| domaindata,e.g.              |        | MSMARCObased |              |     |       |     |         |         |      |     |
| ---------------------------- | ------ | ------------ | ------------ | --- | ----- | --- | ------- | ------- | ---- | --- |
| (Table 2).                   | We see | that         | Promptriever |     |       |     |         |         |      |     |
| performscomparablytoRepLLaMA |        |              |              |     | Model |     | DL19    |         | DL20 | Dev |
| on in-domain                 | tasks  | despite      | addition-    |     |       |     | nDCG@10 | nDCG@10 |      | MRR |
ally having stronger instruction fol- RepLLaMA *74.5 *71.8 *42.5
| lowingperformance. |     |     |     |     | Promptriever |     | *73.2 |     | *72.3 | 42.0 |
| ------------------ | --- | --- | --- | --- | ------------ | --- | ----- | --- | ----- | ---- |
11Itislikelythatacross-encodertrainedonourgenerateddatawouldoutperformFollowIR-7B,however,we
leavethatforfutureworkandfocusonlyondenseretrieversinthiswork.
12Although
standard NQ has a training set, the BEIR NQ version selects a different set of queries and
documentsanddoesn’tprovideacomparabledev/trainset,c.f.<removedforanonymity>.
5

PublishedasaconferencepaperatICLR2025
Table3: OutofdomainperformanceonBEIR(nDCG@10). Promptrieverperformssimilarlyto
RepLLaMAwithoutanyinstructions(Nonecolumn).
However,whengivenapromptweseethat
performanceimprovesforPromptrieverby+1.4pointswhereasRepLLaMAandBM25perform
worse. WethusseethatPromptrieverispromptableduetoitsinstruction-training. Wealsosee
thatitispossibletoselectthebestpromptfrom10devexamplesconsistentlyforPromptriever,as
thedifferencebetweentheSelectedPromptisalmostalwaysthesameastheBestPromptoftheten
promptsevaluated. NotethatnotallBEIRdatasetshavetraining/devsetsandthusSelectedPromptis
leftblankforthem. Bestvalueintherowisbolded(seeAppendixCforstatisticaltests).
|     |     | BM25 |     | RepLLaMA |     | Promptriever |     |
| --- | --- | ---- | --- | -------- | --- | ------------ | --- |
Dataset
|         |      |          | Best      |          | Best   |               | Best   |
| ------- | ---- | -------- | --------- | -------- | ------ | ------------- | ------ |
|         | None | Selected | None      | Selected |        | None Selected |        |
|         |      | Prompt   | Prompt    | Prompt   | Prompt | Prompt        | Prompt |
| DBPedia | 29.9 | 23.3     | 23.3 44.8 | 16.8     | 43.5   | 45.0 45.2     | 45.2   |
tesniart/vedsahtesataD
| FEVER         | 48.1 | 7.4  | 45.3 82.9 | 85.3 | 85.3 | 82.8 82.8 | 82.8 |
| ------------- | ---- | ---- | --------- | ---- | ---- | --------- | ---- |
| FiQA          | 25.1 | 16.9 | 21.9 45.0 | 41.8 | 42.7 | 45.9 46.6 | 46.6 |
| HotpotQA      | 56.9 | 54.6 | 54.6 68.8 | 66.4 | 67.9 | 69.2 69.5 | 69.5 |
| NFCorpus      | 32.1 | 18.0 | 23.5 36.0 | 34.2 | 35.0 | 36.5 36.9 | 36.9 |
| Quora         | 80.4 | 75.3 | 75.3 86.0 | 83.1 | 85.5 | 86.5 88.0 | 88.0 |
| SciFact       | 68.7 | 65.7 | 65.7 75.3 | 75.0 | 75.7 | 75.0 75.9 | 76.3 |
| Arguana       | 36.6 | -    | 36.4 48.6 | -    | 49.0 | 51.8 -    | 56.7 |
| Climate-FEVER | 13.6 | -    | 13.9 29.3 | -    | 30.8 | 27.6 -    | 32.1 |
tesniart/vedoN
| NQ12        | 28.5 | -   | 25.4 63.0 | -   | 62.2 | 61.9 - | 62.6 |
| ----------- | ---- | --- | --------- | --- | ---- | ------ | ---- |
| SCIDOCS     | 15.8 | -   | 14.9 16.1 | -   | 16.7 | 17.3 - | 19.7 |
| TREC-COVID  | 62.3 | -   | 35.9 83.9 | -   | 82.7 | 83.9 - | 84.6 |
| Touche-2020 | 33.1 | -   | 30.8 34.1 | -   | 35.9 | 31.4 - | 32.0 |
| Average     | 40.9 | -   | 35.9 54.9 | -   | 54.8 | 55.0 - | 56.4 |
4.3 OUT-OF-DOMAINRETRIEVAL
We use the BEIR benchmark to measure out-of-domain performance. We do so in two settings:
(1)nopromptsand(2)withprompts. ForoutofdomainperformanceonBEIR(Table3)without
prompts(i.e. theNonecolumn)wealsofindcomparablescores: Promptrieverperformssimilarlyto
| RepLLaMA(Promptrieveraveraging55.0vs. |     |     | 54.9fromRepLLaMA). |     |     |     |     |
| ------------------------------------- | --- | --- | ------------------ | --- | --- | --- | --- |
However, we can also evaluate them with prompts. We use the common approach from the LM
community, which seeks to improve LMs on out-of-domain data by including a textual prompt
attest-time,evenifthepromptissomewhatgeneric,e.g.,“thinkstepbystep"or“I’llgiveyoua
tip"(Kojima et al., 2022; Wei et al., 2022b). We apply this approach to IR by exploring whether
particularpromptsreliablyinduceimprovedretrievalperformanceinPromptriever.
Weusethefollowingsettingsfortestingprompts,followingthestandardsintheLMcommunity;
typicallyonewouldevaluatepromptsforanLMbyfirstusingasmallvalidationset. Wesample10
queriesfromeachofthevalidation(ortrainifthereisnovalidationset)touseastheprompttuning
set. Wealsocreate10genericprompts13thatcouldworkacrossretrievaldatasets.
However,notalloftheBEIRdatasetshavetrain/devdatatosamplevalidationexamplesfromfor
selecting a prompt. We thus show results in two ways (Table 3): (1) when there is a dev set we
selectthebestdevpromptasthetestprompt(SelectedPromptcolumn)andleavethescoreblankfor
datasetswithoutadev/trainset;and(2)takingthebestpromptoftheten(BestPromptcolumn).
WeseeinTable3that,forPromptriever,usingthebestpromptbringssignificantgainstoBEIR14
averageperformance(+1.4nDCG@10;gainsversusnopromptfor12/13datasetsandtiedonthe
last). However,incontrast,promptsfailtobringanygainstotheRepLLaMAorBM25modelswith
-0.1and-5.0nDCGdeltasrespectively. ThuswecanseethatpromptingiseffectiveforPromptriever
butnotforretrievalmodelsusingstandardtraining.
13WeincludethegeneratedpromptsinAppendixA
14IndividualpromptsandtheirscoresoneachBEIRdatasetarefoundinTable11.
6

PublishedasaconferencepaperatICLR2025
Table4:AblationsforinstructionfollowingontheFollowIRandInstructIRdatasets. Theinstruction
provides gains beyond the simple length of the instruction or its distribution. Instruction-
negativesandjointdatabringevenmoregains. Bestvalueisinthecolumnisboldedwhileasterisks
indicatestatisticalsimilaritytothebestscore.
|          |     |     |           |      |        | FollowIR |            |             | InstructIR |         |
| -------- | --- | --- | --------- | ---- | ------ | -------- | ---------- | ----------- | ---------- | ------- |
| Model    |     |     | Robust04  |      | News21 |          | Core17     | Average     | MSMARCO    |         |
|          |     |     | MAP p-MRR |      | nDCG   | p-MRR    | MAP p-MRR  | Score p-MRR | nDCG       | Robust. |
| RepLLaMA |     |     | 24.0      | -8.9 | *24.5  | -1.8     | *20.6 +1.3 | 23.0 -3.1   | 85.7       | 50.2    |
RepeatedQuery 24.6 -9.1 *25.3 -2.6 *21.1 +2.4 23.6 -3.1 85.4 49.2
GenericInstruct 25.5 -7.2 *26.2 -1.7 *21.6 -0.0 *24.4 -3.0 63.1 32.4
SwapInstruct 25.2 -1.9 *27.3 -0.2 *21.1 -0.6 *24.6 -0.9 48.6 27.0
w/Instructions *26.9 +3.8 *29.1 *+5.3 *20.7 +8.0 *25.6 +5.7 *91.9 *63.3
w/InstructionNegatives *29.0 *+9.7 *27.8 *+5.2 *21.9 +11.4 *26.2 +8.8 *91.5 *62.0
Promptriever(Joint) *28.3 *+11.7 *28.5 *+6.4 *21.6 *+15.4 *26.1 *+11.2 *92.1 *63.1
25
| But are | these | best prompt | numbers |     | close | to  |     |     | Model |     |
| ------- | ----- | ----------- | ------- | --- | ----- | --- | --- | --- | ----- | --- |
whatwouldbeachievedinpracticewithasmall tesataD reP stpmorP fo veD dtS BM25
| dev set? | The | Selected | Prompt | column | shows |     |     |     | RepLLaMA |     |
| -------- | --- | -------- | ------ | ------ | ----- | --- | --- | --- | -------- | --- |
20
| score from                                | a practical |       | setting.               | If we | compare |     |     |     | Promptriever |     |
| ----------------------------------------- | ----------- | ----- | ---------------------- | ----- | ------- | --- | --- | --- | ------------ | --- |
| Promptriever’s                            |             | score | for Selected           |       | Prompt  | vs  |     |     |              |     |
| BestPrompt,weseethatthereisverylittledif- |             |       |                        |       |         |     | 15  |     |              |     |
| ferencebetweenthem.                       |             |       | Applyingfew-shotselec- |       |         |     |     |     |              |     |
tionwiththedevsetselectsthebestpromptin
6/7cases,andinthe7thcase(SciFact)itchooses
10
apromptthatisstillalmostonefullpointbet-
| terthanthenopromptsetting. |     |     |     | Incontrast,and |     |     |     |     |     |     |
| -------------------------- | --- | --- | --- | -------------- | --- | --- | --- | --- | --- | --- |
asexpected,BM25isnot“promptable"inany
5
| setting with | performance |     | dropping |     | across the |     |     |     |     |     |
| ------------ | ----------- | --- | -------- | --- | ---------- | --- | --- | --- | --- | --- |
board;RepLLaMA’sperformancedrops(some-
timesdramatically)insixofsevencases.
0
|         |         |     |             |     |          |     | BM25 | RepLLaMA | Promptriever |     |
| ------- | ------- | --- | ----------- | --- | -------- | --- | ---- | -------- | ------------ | --- |
| We also | examine | the | sensitivity | of  | all mod- |     |      |          |              |     |
Model
| els to the     | prompts | (Figure  | 3).        | We  | see that    |        |                |                 |         |          |
| -------------- | ------- | -------- | ---------- | --- | ----------- | ------ | -------------- | --------------- | ------- | -------- |
|                |         |          |            |     |             | Figure | 3: Standard    | dev. of         | NDCG@10 | scores   |
| Promptriever’s |         | variance | to prompts |     | is signifi- |        |                |                 |         |          |
|                |         |          |            |     |             | per    | dataset across | all 10 prompts. | We      | see that |
| cantly less    | than    | that of  | RepLLaMA   |     | (by 44%)    |        |                |                 |         |          |
Promptrieverismuchmorerobusttothephrasing.
andBM25(by77%)whichhaswideswingsdue
| totheeffectofthekeywordmatching. |     |     |     |     | Thissug- |     |     |     |     |     |
| -------------------------------- | --- | --- | --- | --- | -------- | --- | --- | --- | --- | --- |
geststhatPromptrieverismorerobusttotheinputandislesssensitivetotheexactkeywordsbeing
used.
Insummary,thestandardpracticeofinstruction-trainingwithLMscanapplytoinstruction-training
denseretrieversaswell,butonlyifthey,likePromptriever,aretrainedtobesensitivetosuchprompts.
Furthermore,stronggainsareindeedreliablypossibletoachieveviaselectinganaturallanguage
promptusingasmallheld-outevalset,similartohowLMsaretypicallyprompted.
5 ANALYSIS
WeablateseveralnullhypothesesinanefforttobetterunderstandwhichpartsofthePromptriever
trainingrecipecontributemosttoperformancegains. Wetrainalloftheablatedmodelsonadataset
consisting of half standard MS MARCO data and half instruction-based MS MARCO data for
computeandspeedreasons(alsomatchingthenumberoftraininginstancesinRepLLaMA).Results
for all ablations are in Table 4, with the baseline being RepLLaMA, and each row of the table
representingeitheranullhypothesisoradesigndecisionleadingtothefinalPromptrievermodels.
Q1: Isitsimplythelengthofthequery/instructionthatenablesPromptriever’sperformance
gains? Answer: No. We train models with the query repeated to the length of the instruction
7

PublishedasaconferencepaperatICLR2025
Table 5: Comparison of different backbone models on the same Promptriever recipe across MS
MARCO datasets (DL19, DL20, and Dev), BEIR, InstructIR, and FollowIR. We see that our
augmentedinstructiondatasetprovidesgainsformanydifferentbasemodels,indicatingthe
generalityofourapproach. Hyperparametertuningwouldlikelyimprovetheseresultsforother
backbonemodels. SeeAppendixGforadiscussiononBERT-sizedmodels.
MSMARCO BEIR FollowIR InstructIR
BaseModel
DL19 DL20 Dev nDCG w/Prompt Score p-MRR nDCG Robust@10
RepLLaMA 73.2 72.3 42.0 54.9 54.8 23.0 -3.1 85.7 50.2
reveirtpmorP LLaMA2 74.5 71.8 42.5 55.0 56.4 26.1 +11.2 92.1 63.1
Mistralv1 72.9 73.4 42.3 54.4 55.7 25.7 +11.8 90.3 58.8
Llama3.1 73.5 72.9 43.2 55.1 56.5 25.0 +11.3 85.5 41.6
Llama3.1Instruct 72.4 73.6 42.7 55.5 57.2 26.0 +9.8 89.9 57.8
(RepeatQuery)andwithGenericInstructions.15 Increasingthelengthofqueriesresultsinaslight
gainonstandardretrievalperformance,butlittlegaininp-MRR(retrievalsensitivity). Thisincludes
theRepeatQuery(+0.6nDCG/MAP)andGenericInstruction(+1.4nDCG/MAP)baselines.
Q2: Is it the lexical distribution of the instructions (in isolation) that enables these gains?
Answer: Partially. For this we train with the real generated instructions but randomly swap the
instructioneachqueryispairedwith(SwapInstructionsrow). Comparedtothelengthablations,
weseelargergainsinthescore(+1.6)andp-MRR(+2.1)asthemodelhaslearnedthedistribution,
althoughnothowtousethemeffectively.
Q3: Howmuchdoestrainingwithinstructionshelp? Answer: Significantly. Weablatethisby
showingtheresultsoftrainingwithjusttheinstructionsandnoinstruction-negatives(w/Instructions).
Weseeastronggaininp-MRR(+6.6)andafurthergaininstandardretrieval(+1)overSwap.
Q4: Howmuchdoestrainingwithinstruction-negativeshelp? Answer: Significantly. Adding
the instruction negatives on top of w/Instructions gives another large gain in p-MRR (+3.1 over
w/Instructions)andasmallboostinstandardretrievalscores(+0.6nDCG/MAP).Thisalignswith
expectations: instructionnegativesprovideextradataforinstructionsensitivitybutnotnecessarily
forstandardretrievalmetrics.
Q5: DoestrainingadditionallyonMSMARCOhelpbeyondthePromptrievertrainingsetwe
curate? Answer:Yes. Promptriever(Joint)ourfinalmodel,combinesallthestandardMSMARCO
andInstructionMSMARCOdatawhichleadstoanotherlargejumpinp-MRR(+2.4)asitisableto
seemoredata(andinstructions)intraining,i.e. 2xasmuch.
In summary, each step in our final recipe (+w/ Instructions, +w/ Instruction Negatives, +MS
MARCO Jointly) provides value independently, and that value is not due to simple factors like
increasingthelengthofthequeryand/orsurfacelexicalfeaturesoftheinstructions.
5.1 DOESTHISPROCESSWORKFOROTHERMODELS?
TheoriginalRepLLaMAusedLlama2asabackbone,and,tothispointinourpaper,Promptrieverhas
alsousedLlama2asabackboneforfaircomparison.Wealsoadoptthesametraininghyperparameters
asRepLLaMA.Nonetheless,weablatedifferentLMbackbonestoseeifperformanceholdswithout
anyadjustmentstothehyperparametersortrainingrecipe. Whilefurthertuningthelearningrateand
otherparameterswouldlikelyimproveperformance,weseeinTable5thatotherbackbonesprovide
comparableperformance,indicatingthegeneralityofourmethod.
15Weaddagenericretrievalinstructionfromoneof50differentgenericretrievaltaskdescriptionsgenerated
byGPT-4oandClaude-3.5-Sonnet.SeeafulllistinAppendixI.
8

PublishedasaconferencepaperatICLR2025
6 RELATED WORK
6.1 INSTRUCTIONSINRETRIEVAL
TheuseofinstructionsisarelativelynewdevelopmentforIRmodels,asdenseretrievertraining
generallyfocusesonlearningsimilarityfunctionssimilartophrase-levelmatching(Craswelletal.,
2020; Izacard et al., 2021; Wang et al., 2022a). Some of the earliest work on the topic is TART
(Asaietal.,2022)andInstructor(Suetal.,2022)whichusedsimpletaskprefixesduringtraining.
Morerecently,E5-Mistral(Wangetal.,2023),GritLM(Muennighoffetal.,2024),andNV-Retriever
(deSouzaP.Moreiraetal.,2024)scaledupthedatasetandmodelsize. Thesenewermodelstypically
re-usethesameinstructionsetproposedbytheE5-Mistralmodel.16 Ourworkdiffersfromthisby
applyingandevaluatingadaptabilityper-queryratherthanusingadatasetwideprefix.
Ourrobustnessevaluationalsogoesbeyondpriorworks: while,e.g.,Suetal.(2022)testsinstruction
phrasingbychangingoneword,weconsiderabroaderrangeoflength/stylemodifications.
Severalbenchmarkeffortsfocusonexplicitlytestingtheinstructionfollowingabilityofretrievers:
FollowIR(Welleretal.,2024)andInstructIR(Ohetal.,2024). Bothfoundthatexistingbi-encoder
retrievalmodelsfailtouseinstructionsasanLMwould. Ourworkpresentsthefirstbi-encoderthat
achievessignificantlyabove-randomperformanceonthesebenchmarks.
6.2 PROMPTINGLMS
Itisnowthedefacto-standardforLMstotakeandreasonoverinputinstructionsgivenviaprompt-
ing. ThiswasdiscoveredandpopularizedbymodelssuchasInstructGPT(Ouyangetal.,2022),
FLAN(Weietal.,2022a),andT0(Sanhetal.,2022). Importantly,theseworksfoundthatdiversity
oftrainingdatawascrucialtogeneralization. InstructionsarealsooftenincludedinLM’straining
datatoencouragethisbehavior,bothinpre-trainingdata(Soldainietal.,2024;Computer,2023)and
followedbystagesofpost-training(includingfine-tuning/RLGroeneveldetal.(2024)).
Although IR models often use LMs as their base architecture before IR training, little work has
exploredusingstandardLMcapabilitieslikepromptabilityinIR.TheclosestworksincludeGritLM
(Muennighoffetal.,2024)whoattemptedtodoin-contextlearning(ICL)withtheirtrainedretriever
butfoundworseresultsthanzero-shotandpotentiallytherecentBGE-ICL,17 althoughasoflate
September2024,thereisnoassociatedpaperdescribingtheirprocedureortrainingdetails.
6.3 SYNTHETICINSTRUCTIONGENERATION
GeneratingsyntheticdatafortrainingisapopulartechniquegiventhestrongperformanceofLMs.
ThishappensinbothNLP(BenAllaletal.,2024;Adleretal.,2024)aswellasinIR,forgenerating
queriesordocumentsfortraining(Bonifacioetal.,2022;Daietal.,2022;Jeronymoetal.,2023).
WebuilduponanotherlineofworkthatusesLMstocreateinstructionsthatcanbeusedforretrieval
training(Wangetal.,2022b;Lietal.,2023a;Chungetal.,2024),however,weusethisapproachto
generateinstruction-negativepassagesinsteadofstandardpassages.
7 CONCLUSION
Wepresentedthefirstzero-shotpromptableretriever,Promptriever,trainedfromanewinstruction-
basedretrievaldatasetbasedonMSMARCO.ExperimentsshowthatPromptrievernotonlyperforms
wellonthestandardretrievaltask,butalsofollowsinstructionsmoreeffectivelythanpriorwork,
adaptingitsnotionofrelevanceper-query. Overall,thisshowsthattechniquesdiscoveredintheLM
community,suchasprompting,canbeextendedtodenseretrieversaswell. Wehopethiswillinspire
jointresearchbetweenthetwocommunitiesandfurtherenableretrieversthatcanadaptonthefly.
16Youcanfindalistofallthe“instructions”atthisurl.
17https://huggingface.co/BAAI/bge-en-icl
9

PublishedasaconferencepaperatICLR2025
8 LIMITATIONS
AlthoughPromptrieverintroducesper-instancepromptingtoretrievalmodels,therearemanyaspects
ofpromptingwithLMsthatwedidnotexploreinthiswork. Forexample,futureworkcouldlook
atin-contextlearning: canretrievalmodelsbepromptedwithafewexamplesexplicitlyversusjust
imperative requests? We leave this to future work to explore and hope to see a wide variety of
promptingtechniquesfromtheLMcommunityappliedtoretrievalmodels.
Wealsonotethat,similartolanguagemodels,itisoftenunclearwhysomeIRpromptsperformbetter
thanothers. Languagemodelshavebecomemorerobusttodifferentpromptsovertime,andwehope
thatfutureworkwillcontinuetoimprovethisabilityforretrievalmodels.
Finally,aswithanyLM-generateddata,itispossiblethatthereareerrors,pernicioussocialbiases,
and/orincorrectpiecesofinformationinthegeneratedpassagesandinstructions.Althoughweapplied
some(probabilistic)correctnessfiltersandconductedquantitativeandqualitativeexplorations,itis
stillpossiblethatunintendedcharacteristicsslippedthrough. Whileexperimentsdemonstratethat
trainingonourcorpusimprovesperformance,furtherauditsofourcorpus(andretrievaltrainingsets
morebroadly)wouldbeappropriateanduseful.
9 ACKNOWLEDGMENTS
WethankthemachinelearningteamatSamayaAIforthehelpfuldiscussionsandfeedback. OWis
supportedbyaNSFGRFPfellowship.
REFERENCES
Bo Adler, Niket Agarwal, Ashwath Aithal, Dong H Anh, Pallab Bhattacharya, Annika Brundyn,
JaredCasper,BryanCatanzaro,SharonClay,JonathanCohen,etal. Nemotron-4340btechnical
report. arXivpreprintarXiv:2406.11704,2024.
AI@Meta. Llama3modelcard. 2024. URLhttps://github.com/meta-llama/llama3/blob/
main/MODEL_CARD.md.
AkariAsai,TimoSchick,PatrickLewis,XilunChen,GautierIzacard,SebastianRiedel,HannanehHa-
jishirzi,andWen-tauYih. Task-awareretrievalwithinstructions. arXivpreprintarXiv:2211.09260,
2022.
ParishadBehnamGhader,VaibhavAdlakha,MariusMosbach,DzmitryBahdanau,NicolasChapados,
andSivaReddy. Llm2vec: Largelanguagemodelsaresecretlypowerfultextencoders. arXiv
preprintarXiv:2404.05961,2024.
LoubnaBenAllal,AntonLozhkov,GuilhermePenedo,ThomasWolf,andLeandrovonWerra. Cos-
mopedia,2024. URLhttps://huggingface.co/datasets/HuggingFaceTB/cosmopedia.
LuizBonifacio,HugoAbonizio,MarziehFadaee,andRodrigoNogueira. Inpars: Dataaugmentation
forinformationretrievalusinglargelanguagemodels. arXivpreprintarXiv:2202.05144,2022.
HyungWonChung,LeHou,ShayneLongpre,BarretZoph,YiTay,WilliamFedus,YunxuanLi,
XuezhiWang,MostafaDehghani,SiddharthaBrahma,etal.Scalinginstruction-finetunedlanguage
models. JournalofMachineLearningResearch,25(70):1–53,2024.
TogetherComputer. Redpajama: anopendatasetfortraininglargelanguagemodels,2023. URL
https://github.com/togethercomputer/RedPajama-Data.
NickCraswell,BhaskarMitra,EmineYilmaz,DanielCampos,andEllenMVoorhees. Overviewof
thetrec2019deeplearningtrack. arXivpreprintarXiv:2003.07820,2020.
ZhuyunDai, VincentYZhao, JiMa, YiLuan, JianmoNi, JingLu, AntonBakalov, KelvinGuu,
KeithBHall,andMing-WeiChang. Promptagator: Few-shotdenseretrievalfrom8examples.
arXivpreprintarXiv:2209.11755,2022.
10

PublishedasaconferencepaperatICLR2025
GabrieldeSouzaP.Moreira,RadekOsmulski,MengyaoXu,RonayAk,BenediktSchifferer,and
Even Oldridge. Nv-retriever: Improving text embedding models with effective hard-negative
mining,2024. URLhttps://arxiv.org/abs/2407.15831.
LuyuGao,XueguangMa,JimmyLin,andJamieCallan. Tevatron: Anefficientandflexibletoolkit
fordenseretrieval. arXivpreprintarXiv:2203.05765,2022.
Dirk Groeneveld, Iz Beltagy, Pete Walsh, Akshita Bhagia, Rodney Kinney, Oyvind Tafjord,
Ananya Harsh Jha, Hamish Ivison, Ian Magnusson, Yizhong Wang, et al. Olmo: Accelerat-
ingthescienceoflanguagemodels. arXivpreprintarXiv:2402.00838,2024.
Gautier Izacard, Mathilde Caron, Lucas Hosseini, Sebastian Riedel, Piotr Bojanowski, Armand
Joulin,andEdouardGrave. Unsuperviseddenseinformationretrievalwithcontrastivelearning.
arXivpreprintarXiv:2112.09118,2021.
VitorJeronymo,LuizBonifacio,HugoAbonizio,MarziehFadaee,RobertoLotufo,JakubZavrel,
and Rodrigo Nogueira. Inpars-v2: Large language models as efficient dataset generators for
informationretrieval. arXivpreprintarXiv:2301.01820,2023.
Albert Qiaochu Jiang, Alexandre Sablayrolles, Arthur Mensch, Chris Bamford, Devendra Singh
Chaplot, Diego de Las Casas, Florian Bressand, Gianna Lengyel, Guillaume Lample, Lucile
Saulnier,L’elioRenardLavaud,Marie-AnneLachaux,PierreStock,TevenLeScao,ThibautLavril,
ThomasWang, TimothéeLacroix, andWilliamElSayed. Mistral7b. ArXiv, abs/2310.06825,
2023. URLhttps://api.semanticscholar.org/CorpusID:263830494.
TakeshiKojima,ShixiangShaneGu,MachelReid,YutakaMatsuo,andYusukeIwasawa. Large
languagemodelsarezero-shotreasoners. Advancesinneuralinformationprocessingsystems,35:
22199–22213,2022.
JinhyukLee,ZhuyunDai,XiaoqiRen,BlairChen,DanielCer,JeremyRCole,KaiHui,Michael
Boratko,RajviKapadia,WenDing,etal. Gecko: Versatiletextembeddingsdistilledfromlarge
languagemodels. arXivpreprintarXiv:2403.20327,2024.
XianLi,PingYu,ChuntingZhou,TimoSchick,LukeZettlemoyer,OmerLevy,JasonWeston,and
MikeLewis. Self-alignmentwithinstructionbacktranslation. arXivpreprintarXiv:2308.06259,
2023a.
ZehanLi,XinZhang,YanzhaoZhang,DingkunLong,PengjunXie,andMeishanZhang. Towards
generaltextembeddingswithmulti-stagecontrastivelearning. arXivpreprintarXiv:2308.03281,
2023b.
XingHanLù. Bm25s: Ordersofmagnitudefasterlexicalsearchviaeagersparsescoring. arXiv
preprintarXiv:2407.03618,2024.
XueguangMa,LiangWang,NanYang,FuruWei,andJimmyLin. Fine-tuningllamaformulti-stage
textretrieval. arXivpreprintarXiv:2310.08319,2023.
LukeMerrick,DanmeiXu,GauravNuti,andDanielCampos. Arctic-embed: Scalable,efficient,and
accuratetextembeddingmodels. arXivpreprintarXiv:2405.05374,2024.
NiklasMuennighoff,NouamaneTazi,LoïcMagne,andNilsReimers. Mteb:Massivetextembedding
benchmark. arXivpreprintarXiv:2210.07316,2022.
NiklasMuennighoff,HongjinSu,LiangWang,NanYang,FuruWei,TaoYu,AmanpreetSingh,and
DouweKiela. Generativerepresentationalinstructiontuning. arXivpreprintarXiv:2402.09906,
2024.
Tri Nguyen, Mir Rosenberg, Xia Song, Jianfeng Gao, Saurabh Tiwary, Rangan Majumder, and
Li Deng. MS MARCO: A human generated machine reading comprehension dataset. CoRR,
abs/1611.09268,2016. URLhttp://arxiv.org/abs/1611.09268.
RodrigoNogueira,ZhiyingJiang,andJimmyLin. Documentrankingwithapretrainedsequence-to-
sequencemodel. arXivpreprintarXiv:2003.06713,2020.
11

PublishedasaconferencepaperatICLR2025
HanseokOh,HyunjiLee,SeonghyeonYe,HaebinShin,HansolJang,ChangwookJun,andMinjoon
Seo. Instructir: Abenchmarkforinstructionfollowingofinformationretrievalmodels. arXiv
preprintarXiv:2402.14334,2024.
LongOuyang,JeffreyWu,XuJiang,DiogoAlmeida,CarrollWainwright,PamelaMishkin,Chong
Zhang, Sandhini Agarwal, Katarina Slama, Alex Ray, John Schulman, Jacob Hilton, Fraser
Kelton, Luke Miller, Maddie Simens, Amanda Askell, Peter Welinder, Paul F Christiano, Jan
Leike,andRyanLowe. Traininglanguagemodelstofollowinstructionswithhumanfeedback.
In S. Koyejo, S. Mohamed, A. Agarwal, D. Belgrave, K. Cho, and A. Oh (eds.), Advances
in Neural Information Processing Systems, volume 35, pp. 27730–27744. Curran Associates,
Inc., 2022. URL https://proceedings.neurips.cc/paper_files/paper/2022/file/
b1efde53be364a73914f58805a001731-Paper-Conference.pdf.
StephenERobertson,SteveWalker,SusanJones,MichelineMHancock-Beaulieu,MikeGatford,
etal. Okapiattrec-3. NistSpecialPublicationSp,109:109,1995.
VictorSanh,AlbertWebson,ColinRaffel,StephenBach,LintangSutawika,ZaidAlyafeai,Antoine
Chaffin,ArnaudStiegler,ArunRaja,MananDey,MSaifulBari,CanwenXu,UrmishThakker,
ShanyaSharmaSharma,ElizaSzczechla,TaewoonKim,GunjanChhablani,NihalNayak,De-
bajyotiDatta,JonathanChang,MikeTian-JianJiang,HanWang,MatteoManica,ShengShen,
ZhengXinYong,HarshitPandey,RachelBawden,ThomasWang,TrishalaNeeraj,JosRozen,
AbheeshtSharma,AndreaSantilli,ThibaultFevry,JasonAlanFries,RyanTeehan,TevenLeScao,
StellaBiderman,LeoGao,ThomasWolf,andAlexanderMRush. Multitaskpromptedtraining
enableszero-shottaskgeneralization. InInternationalConferenceonLearningRepresentations,
2022. URLhttps://openreview.net/forum?id=9Vrb9D0WI4.
IanSoboroff. Overviewoftrec2021. In30thTextREtrievalConference.Gaithersburg,Maryland,
2021.
LucaSoldaini,RodneyKinney,AkshitaBhagia,DustinSchwenk,DavidAtkinson,RussellAuthur,
BenBogin,KhyathiChandu,JenniferDumas,YanaiElazar,etal. Dolma: Anopencorpusofthree
trilliontokensforlanguagemodelpretrainingresearch. arXivpreprintarXiv:2402.00159,2024.
Hongjin Su, Weijia Shi, Jungo Kasai, Yizhong Wang, Yushi Hu, Mari Ostendorf, Wen-tau Yih,
NoahA.Smith,LukeZettlemoyer,andTaoYu. Oneembedder,anytask: Instruction-finetuned
textembeddings. 2022. URLhttps://arxiv.org/abs/2212.09741.
NandanThakur,NilsReimers,AndreasRücklé,AbhishekSrivastava,andIrynaGurevych. Beir: A
heterogenousbenchmarkforzero-shotevaluationofinformationretrievalmodels. arXivpreprint
arXiv:2104.08663,2021.
HugoTouvron,ThibautLavril,GautierIzacard,XavierMartinet,Marie-AnneLachaux,Timothée
Lacroix, BaptisteRozière, NamanGoyal, EricHambro, FaisalAzhar, etal. Llama: Openand
efficientfoundationlanguagemodels. arXivpreprintarXiv:2302.13971,2023a.
HugoTouvron,LouisMartin,KevinStone,PeterAlbert,AmjadAlmahairi,YasmineBabaei,Nikolay
Bashlykov,SoumyaBatra,PrajjwalBhargava,ShrutiBhosale,etal. Llama2: Openfoundation
andfine-tunedchatmodels. arXivpreprintarXiv:2307.09288,2023b.
LiangWang,NanYang,XiaolongHuang,BinxingJiao,LinjunYang,DaxinJiang,RanganMajumder,
andFuruWei. Textembeddingsbyweakly-supervisedcontrastivepre-training. arXivpreprint
arXiv:2212.03533,2022a.
LiangWang,NanYang,XiaolongHuang,LinjunYang,RanganMajumder,andFuruWei. Improving
textembeddingswithlargelanguagemodels. arXivpreprintarXiv:2401.00368,2023.
YizhongWang,YeganehKordi,SwaroopMishra,AlisaLiu,NoahASmith,DanielKhashabi,and
Hannaneh Hajishirzi. Self-instruct: Aligning language model with self generated instructions.
arXivpreprintarXiv:2212.10560,2022b.
JasonWei,MaartenBosma,VincentY.Zhao,KelvinGuu,AdamsWeiYu,BrianLester,NanDu,
AndrewM.Dai,andQuocV.Le. Finetunedlanguagemodelsarezero-shotlearners,2022a.
12

PublishedasaconferencepaperatICLR2025
JasonWei,XuezhiWang,DaleSchuurmans,MaartenBosma,FeiXia,EdChi,QuocVLe,Denny
Zhou,etal. Chain-of-thoughtpromptingelicitsreasoninginlargelanguagemodels. Advancesin
neuralinformationprocessingsystems,35:24824–24837,2022b.
OrionWeller,BenjaminChang,SeanMacAvaney,KyleLo,ArmanCohan,BenjaminVanDurme,
DawnLawrie,andLucaSoldaini. Followir: Evaluatingandteachinginformationretrievalmodels
tofollowinstructions. arXivpreprintarXiv:2403.15246,2024.
A RETRIEVAL PROMPTS
InTable10weshowtheretrievalpromptsandtheirscoresperdatasetinTable11.
B CROSS-ENCODER VS BI-ENCODER EFFICIENCY
Cross-encoderscomputefullattentionovercombinationsofqueriesanddocumentsatinferencetime
andareO(Q*D)whereQisthenumberofqueriesandDisthenumberofdocuments.
Ontheotherhand,denseretrievers(a.k.a. bi-encoders)likePromptrievercomputeattentionover
queriesanddocumentsindividually,andestimaterelevanceusingacheapdotproduct,yieldingO(Q
+D+dot-product). Thedotproductisnegligible,andisoftenimplementedwithavectordatabasein
practice.
Thus,wecanseethatcross-encodersaresignificantlymoreexpensivetouse,whichiswhymost
retrievalsystemsuseafirststagepasswithabi-encoderandthenfollowthatupwithacross-encoder.
C STATISTICAL SIMILARITY TESTS ON BEIR
Due to the complexity of the figure, we include statistical tests here. We compare two groups:
Promptriever(noprompt)vsRepLLaMA(noprompt)andPromptriever(prompt)vsPromptriever
(nopromt).
(1) Promptriever vs RepLLaMA (no prompts): Promptriever is significantly better for Arguana,
HotpotQA, Quora, SciDocs, while RepLLaMA is significantly better at NQ and Climate-Fever
(p<0.05forall),andtheyarestatisticallysimilarfortheothers.
(2) Promptriever with prompts vs without them: prompts are significantly better for Arguana,
HotpotQA,Quora,SciDocs,andClimate-Fever(p<0.05),whileallothersarestatisticallysimilar.
D INSTRUCTION DEFINITION
Mostqueriesinretrievalareshortandoftenambiguous. Inpracticethecrucialdifferencebetween
aqueryandaninstructionisthataninstructionprovidesmorespecificinput. Thiscouldinclude
specificsforwhatdefinesrelevance,specificsthatdefinenot-relevance,oranyotherbackground/
additionalinformation. Instructionsarealsotypicallylongerthanqueries,althoughitispossible
tohaveashortinstructionoralongquery. Asbothprovideuserinputforwhattheyaresearching
for,theexactboundarycanbehardtodefineexactly. However,forcurrentevaluationdatasets,it
fairlyeasytodisambiguate: queriesfromNQandMSMARCOarestandardandshort,whilethose
FollowIRandInstructIRprovidemanyofthefeaturesdefinedearlierandaresignificantlylonger.
E HYPERPARAMETER DETAILS
WeusethefollowinghyperparametersasgivenbytheauthorsoftheRepLLaMAontheirGithub
page using Tevatron (Gao et al., 2022). This is using meta-llama/Llama-2-7b-hf with lora r
32,loramodulesq_proj,k_proj,v_proj,o_proj,down_proj,up_proj,gate_proj,enabledbfloat16,
using eos pooling, using normalization, a temperature of 0.01, learning rate of 1e-4, one epoch,
passagelength256,100warmupsteps,atraingroupsizeof16,andaneffectivebatchsizeof128
(4 GPUs, 8 per device with a 4 accumulation steps). We differ from the original paper by using
13

PublishedasaconferencepaperatICLR2025
querymaxlength304toaccountforthelongerinstructions(previouslysetto32inRepLLaMA).
Trainingtakesapproximately2dayson8x40GBA100sfortheablationrunsand4daysforthefull
run. Inferencetakesuptofourhoursperdatasetonthe8xclusterusing512lengthparametersfor
queryandpassages.
| F   | GENERATION | PROMPTS |     |     |     |     |     |     |     |     |
| --- | ---------- | ------- | --- | --- | --- | --- | --- | --- | --- | --- |
Weincludethepromptsusedtogeneratethedata. Forthesystemprompts,weusedPrompt6,forthe
instructiongenerationweusedPrompt4,andfortheinstructionnegativesweusedPrompt5.
| G   | BERT-SIZE | MODEL | BACKBONES |     |     |     |     |     |     |     |
| --- | --------- | ----- | --------- | --- | --- | --- | --- | --- | --- | --- |
WenotethatwealsotriedtrainingBERT-sizedbackboneswiththePromptrieverdata. However,we
~
foundsignificantlyreducedperformance(e.g. 30MRRonMSMARCOdev).Wetriedtrainingwith
theexactsametechnique(LoRA,EOStokenpooling)aswellasmorestandardmethodsforBERT
(CLStoken,fullfine-tuning)andhadsimilarresults. Itispossiblethatthesesmallermodelsarenot
abletounderstandinstructionsaseffectively,afindingwhichismirroredintheLMcommunity.
| H   | MORE QUALITATIVE |     | EXAMPLES |     |     |     |     |     |     |     |
| --- | ---------------- | --- | -------- | --- | --- | --- | --- | --- | --- | --- |
H.1 FOLLOWIREXAMPLE
Query: Identifydocumentsthatdiscussthereasonsforautomobilerecalls.
Instruction: Arelevantdocumentwillspecifymajororminorreasonsforautomobilerecallsbycar
| manufacturers. | Documentsthatdiscusstruckrecallsarenotrelevant. |     |     |     |     |     |     |     |     |     |
| -------------- | ----------------------------------------------- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
RETRIEVEDPASSAGESCOMPARINGPROMPTRIEVERANDREPLLAMA
| Promptriever: |     | Rank1(relevance=True): |     |     |     |     |     |     |     |     |
| ------------- | --- | ---------------------- | --- | --- | --- | --- | --- | --- | --- | --- |
YOURWHEELS:DEFECTINVESTIGATIONSATRECORDHIGHIN’89
|     | February                                            | 1, 1990,  | Thursday, | Home      | Edition   |       | The auto | safety              | tally for | 1989 has |
| --- | --------------------------------------------------- | --------- | --------- | --------- | --------- | ----- | -------- | ------------------- | --------- | -------- |
|     | comein,andthenewsisn’tparticularlygoodforconsumers. |           |           |           |           |       |          | Lastyear,6.7million |           |          |
|     | vehicles,                                           | including | cars made | by        | virtually | every | major    | producer,           | were      | recalled |
|     | by manufacturers                                    |           | because   | of safety | defects   | that  | needed   | correcting.         | The       | recalls  |
included1.7millionGeneralMotorscarswithstickingthrottles,1.4millionFord
Escortswithdefectiveseats,500,000Chryslerswithrustyfront-suspensionparts,
and408,000ToyotaCamryswithproblemseatbelts.TheNationalHighwayTraffic
SafetyAdministrationboastedthatitopened213defectinvestigationsduring1989,
|     | arecordinthe20-yearhistoryoftheagency. |               |     |      |                | Ofthe6.7millionrecalls,theagency |         |           |     |            |
| --- | -------------------------------------- | ------------- | --- | ---- | -------------- | -------------------------------- | ------- | --------- | --- | ---------- |
|     | claims that                            | it influenced |     | 77%. | The statistics |                                  | beg the | question: | Are | cars safer |
becausemoreinvestigationsarebeingrun,ordocarshavemoredefectsthanever?
Thenumbers,criticssay,showthepublicisatrisk...
Rank2(relevance=False):
|     | JAPAN | WARNING | ITS | CAR | MAKERS | AFTER | RASH | OF  | DEFECTS, | RE- |
| --- | ----- | ------- | --- | --- | ------ | ----- | ---- | --- | -------- | --- |
CALLS
|     | December | 5, 1989, | Tuesday, | P.M. | Final | Transport | Ministry |     | officials | will ask |
| --- | -------- | -------- | -------- | ---- | ----- | --------- | -------- | --- | --------- | -------- |
Japanesecarmakerstopaymoreattentiontodesignandqualitycontrolbecauseof
defectsfoundinnewandremodeledcarsrecently,aministryofficialsaidtoday.
Theofficialswillissuetherequestataregularlyscheduledmeetingwiththeheads
|     | of the Japan | Automobile |     | Manufacturers |     | Assn. | on Thursday. |     | Since April | 1, 11 |
| --- | ------------ | ---------- | --- | ------------- | --- | ----- | ------------ | --- | ----------- | ----- |
Japanesemodelsand14foreignmodelshavebeenrecalledbecauseofdefects.This
compareswith15Japanesemodelsand24foreignmodelsintheyeartoMarch31,
1989...
14

PublishedasaconferencepaperatICLR2025
RepLLaMA: Rank1(relevance=False):
full-sizedGMpickuptrucksandcab-chassisequippedwithfueltanksmounted
| outboardoftheframerailscontainadefectrelatedtomotorvehiclesafety. |     |     |     |     |     |     | Atthe |
| ----------------------------------------------------------------- | --- | --- | --- | --- | --- | --- | ----- |
sametime,hereleasedacomprehensiveEngineeringAnalysis(EA)Reportthat
| explainsindetailthebasisforthedecision.         |                                         |     |     | AsummaryofthatReportissetoutin |                    |               |     |
| ----------------------------------------------- | --------------------------------------- | --- | --- | ------------------------------ | ------------------ | ------------- | --- |
| theFederalRegisternoticeannouncingthatdecision. |                                         |     |     | 59Fed.                         | Reg.               | 54025(October |     |
| 27,1994).                                       | TheEAReportmakestwofundamentalfindings. |     |     |                                | First,onthebasisof |               |     |
statisticalanalysis,laboratorycrashtests,andareviewofavailableinformation
concerningspecificreal-worldcrashes,theEAReportdemonstratesthattheGM
pickupsatissueexhibitanincreasedriskoffireinside-impactcrashescomparedto
similarvehicles(specifically,tocontemporaryFordandDodgefull-sizedpickups).
Second,primarilyonthe...
Rank2(relevance=False):
| or type of | crash? If so, how | relevant? | 4.  | Are the extent | of a | manufacturer’s |     |
| ---------- | ----------------- | --------- | --- | -------------- | ---- | -------------- | --- |
awarenessofapotentialorongoingsafetyrisk,andtheextentofamanufacturer’s
| efforts to | avoid that risk, relevant | to  | the issue | of whether | an unreasonable |     | risk |
| ---------- | ------------------------- | --- | --------- | ---------- | --------------- | --- | ---- |
exists? Isamanufacturer’sfailuretoimplementmeasurestomitigateoreliminate
| anincreasedsafetyriskrelevanttothatissue? |     |     |     | 5. Whatweightshouldbegiven |     |     |     |
| ----------------------------------------- | --- | --- | --- | -------------------------- | --- | --- | --- |
totheforegoingthreefactors,andanyotherrelevantfactors,indecidingwhether
| a vehicle | contains a defect related                                         | to  | motor | vehicle safety? | Other | Information |     |
| --------- | ----------------------------------------------------------------- | --- | ----- | --------------- | ----- | ----------- | --- |
| Sought1.  | Additionalinformationconcerningpost-crashfiresinreal-worldcrashes |     |       |                 |       |             |     |
involvingtheGMvehiclescoveredbythisinvestigationandcomparablevehicles.
2. AdditionalinformationregardingGM’sawarenessofanincreasedriskofpost-
crashfiresduetotheplacementofthefueltanksoutsideoftheframerailsofthese
| trucks,bothatthetimeoftheirdesignandafterward. |     |     |     | ... |     |     |     |
| ---------------------------------------------- | --- | --- | --- | --- | --- | --- | --- |
Comment: YoucanseethatRepLLaMAincorrectlyfocusesontrucks,despitetheinstructionto
avoidthem. WhereasPromptrievercorrectlyfocusedonsedanrecalls.
H.2 INSTRUCTIREXAMPLE
Query: whatisthetemperatureinflagstaffaz?
Instruction: AsaremoteworkerconsideringrelocatingtoFlagstaff,AZ,I’mtryingtocomparethe
temperaturetherewithothercitiesonmylist. Mysearchwouldfocusonacomprehensiveweather
toolthatcanshowmeacomparisonoftemperaturesalongwithcost-of-livingindices.
RETRIEVEDPASSAGESCOMPARINGPROMPTRIEVERANDREPLLAMA
Promptriever: Rank1(relevance=True):
ComparisonToolforClimateandLivingExpensesinVariousCities-February
InsightsLocatedatahigherelevation,thiscityexperiencescoolertemperatures
| inthewintermonths. | Thedaytimehighsaroundthistimeofyearaverageat7°C |     |     |     |     |     |     |
| ------------------ | ----------------------------------------------- | --- | --- | --- | --- | --- | --- |
(45°F),whilethenocturnallowsoftenplungetoanaverageof-10°C(14°F).This
intuitivetoolnotonlydisplaysthesetemperaturestatisticsforFebruarybutalso
allowsyoutojuxtaposethisdatawiththatofotherpotentialcitiesonyourradar.
| Furthermore, | foracomprehensiveoverview, |     |     | thiscomparisoncanbetailoredto |     |     |     |
| ------------ | -------------------------- | --- | --- | ----------------------------- | --- | --- | --- |
includelivingexpenses,presentingyouwithadual-axischartthatinformsboth
yourbudgetingandclimatepreferences.
Rank2(relevance=False):
Realestatestakeholdersnowhaveaccesstoacomprehensiveclimateprofileof
| Flagstaff,whichincludesanarrayofmeticulouslycraftedweathercharts. |     |     |     |     |     |     | Each |
| ----------------------------------------------------------------- | --- | --- | --- | --- | --- | --- | ---- |
monthisrepresented,showcasingtemperaturetrendswithpeaksinJulyaveraging
acomfortable78.2°F,whilethewintersintroduceacoolerdemeanorwithDecem-
ber’saveragerestingat29.6°F.Precipitationpeaksarehighlighted,withOctober
15

PublishedasaconferencepaperatICLR2025
Table6: Word-leveldatasetstatisticsfromaugmentingMSMARCOtrainwithinstructions. The
datasetwasgeneratedfromacross-productoflengthformatandfeatureswitheachsubsethaving
~
approximately122kinstances(andthetotaldataset 490k). Themeannumbersareroundedtothe
nearestdigit. WeseethatLlama370Bgenerallyfollowedthelengthdescription.
|     |     | Category |     | Min | Mean |     | Max |     |     |
| --- | --- | -------- | --- | --- | ---- | --- | --- | --- | --- |
|     |     | None     |     |     | 16   | 101 | 369 |     |     |
erutaeFelytS
|                                                    |              | Negation   |                                               |       | 18     | 98                      | 363     |            |     |
| -------------------------------------------------- | ------------ | ---------- | --------------------------------------------- | ----- | ------ | ----------------------- | ------- | ---------- | --- |
|                                                    |              | Background |                                               |       | 19     | 108                     | 340     |            |     |
|                                                    |              | Persona    |                                               |       | 26     | 106                     | 374     |            |     |
|                                                    | tamroFhtgneL | Short      |                                               |       | 16     | 40                      | 84      |            |     |
|                                                    |              | Medium     |                                               |       | 43     | 84                      | 154     |            |     |
|                                                    |              | Long       |                                               |       | 43     | 107                     | 185     |            |     |
|                                                    |              | VeryLong   |                                               |       | 96     | 181                     | 374     |            |     |
|                                                    |              | All        |                                               |       | 16     | 103                     | 374     |            |     |
| notableforitsrainfall,averagingapproximately1inch. |              |            |                                               |       |        | Thisclimatologicalsnap- |         |            |     |
| shot, accumulated                                  | from         | several    | years’                                        | data, | caters | to those                | placing | importance |     |
| onhistoricalweatherpatterns.                       |              |            | Additionally,curatedfortheenvironmentallycon- |       |        |                         |         |            |     |
sciousbuyer,thehealthofFlagstaff’satmosphereisquantified,boastinglowerair
| pollutantlevelscomparedtostateandnationalbenchmarks. |     |     |     |     |     |     | Thisinformationhas |     |     |
| ---------------------------------------------------- | --- | --- | --- | --- | --- | --- | ------------------ | --- | --- |
beenvisualizedinaneasilydigestibleformat,readyforintegrationinlistingsto
guideclientelethroughtheperksofFlagstaff’sseasonalrhythms
RepLLaMA: Notethattherelevantdocisnotinthetop5.
Rank1(relevance=False):
Realestatestakeholdersnowhaveaccesstoacomprehensiveclimateprofileof
Flagstaff...(sameasabove)
Rank2(relevance=False):
| During the | past five   | years, | Flagstaff, | Arizona, |     | has experienced |       | varied climatic |     |
| ---------- | ----------- | ------ | ---------- | -------- | --- | --------------- | ----- | --------------- | --- |
| conditions | pivotal for | any    | adventurer | planning |     | an extended     | stay. | In July,        | the |
warmestmonth,averagehighshaveconsistentlyhoveredaround78degrees,based
| on an in-depth | analysis | of  | the half-decade’s |     | climatic | patterns. |     | Conversely, | the |
| -------------- | -------- | --- | ----------------- | --- | -------- | --------- | --- | ----------- | --- |
wintermonthofDecemberhaspresentedaveragelowsapproximating30degrees,
| acrucialdetailforthecorrectselectionofthermalgear. |     |     |     |     |     | Historicalmonthlyrainfall |     |     |     |
| -------------------------------------------------- | --- | --- | --- | --- | --- | ------------------------- | --- | --- | --- |
peakedinOctober,averaging1inch-astatisticthatcouldinfluencethetiming
| ofhikesandoutdooractivities. |     |     | Thisdata,sourcedfromacomprehensivereview |     |     |     |     |     |     |
| ---------------------------- | --- | --- | ---------------------------------------- | --- | --- | --- | --- | --- | --- |
ofmeteorologicalrecords,ensuresboththeprecisionandreliabilityrequiredfor
| a meticulous | travel | guide. | The air | quality | index | and pollution |     | levels, although |     |
| ------------ | ------ | ------ | ------- | ------- | ----- | ------------- | --- | ---------------- | --- |
nottheprimaryconcernfortemperatureplanning,remainsignificantlybetterthan
statewideandnationalaverages,thusassuringabreathable,pristineenvironment
conducivetoexploration.
Comment: RepLLaMAwasnotabletoidentifythemostimportantpieceoftheinstruction, the
comprehensiveweathertool. Promptrieveridentifieditandplaceditasrank1whereasitdidnot
makethetop5forRepLLaMA.
I GENERIC INSTRUCTIONS
WeshowthegenericinstructionsgiventothemodelsinTable7. Theseweregeneratedbyprompting
GPT-4oandClaude-3.5-Sonnetforgenericretrievaldescriptions.
16

PublishedasaconferencepaperatICLR2025
PromptforInstructionGeneration
##InputData
IhavethefollowingqueryandREL_DOCS_NUM_FILL_MEdocumentswhichhavebeen
markedasrelevantandNON_REL_DOCS_NUM_FILL_MEwhicharenon-relevant.
Query: QUERY_FILL_ME
POS_DOC_FILL_ME
NEG_DOC_FILL_ME
##Yourtask
Ineedyoutocomeupwithaninstructionthatcanbeappendedontotheendofthisquery
thatwillmakeonlyonerelevantdocumentandmakeallotherdocuments(including
previouslyrelevantdocs)non-relevant. Youcanchoosewhichdocumentwillstayrelevant
to the new instruction, by writing an instruction that applies to only one of the relevant
documents(youchoose). Thisadditionalinstructionshouldprovideatestforstrongfrontier
languagemodelstodetermineiftheycanfollowinstructions. Triplecheckyourworktobe
certainthatthechosendocumentisstillrelevantandthattheothersarenon-relevant–ifyou
messupyouwillbefired. Donotgiveawaytheanswerintheinstruction!
For this example, please generate the instruction to be LENGTH_FORMAT_FILL_ME.
In the instructions, provide detailed specifics for what makes a document relevant.
Rememberthatthiscriteriashouldmaketheonedocumentrelevantandallothersirrelevant.
Alsobesurethattheinstructionisgenericanddoesnotcontaintheanswertothequery.
OutputtheresponseinJSONformonlywithnoothertext,withthekeys,“instruction”(str),
“relevant_docs”(onedocumentidthatisthefirstdoc,e.g. “[2]”)and“non-relevant_docs”(all
otherdocumentids,e.g. “[1,3,...]”).
##Youroutput(JSONonly):
Figure4: PromptforInstructionGeneration
17

PublishedasaconferencepaperatICLR2025
Promptforinstructionnegatives
Generatethree 100wordpassages(andexplanations)thatdirectlyanswerthequerybutdo
notprovideavaliddocumentaccordingtothespecificquery. Thengenerateonepassagethat
matchesboth. Makeitobvioustoareaderwhichonesarewhich.
Query: QUERY_FILL_ME
SpecificQuery: INSTRUCTION_FILL_ME
Rememberyourgoalistogeneratearelevantdocument(MSMARCOstyle,withpassage
andtitle)forthequerybutanon-relevantdocumentforthespecificquery. Youshould
generateonlyfactualinformation.
To be crystal clear, your generated documents should have related information about
"QUERY_FILL_ME". However,thesegenerateddocumentsshouldnotberelevanttothe
specificquery. Asexamples,theymayomitcrucialinformationthatisneededforthespecific
query,ifthequeryisambiguousitmayuseanalternativemeaning,oritmayspecifically
mentionelementsthataresaidtobenon-relevant.
Youshouldalsogenerateanexplanationwiththecategoryitisandasuccinctreason. The
tagsare"differentinterpretation","omission","mentionnon-relevantflag"or"none"forthe
relevanttoboth. E.g. "omission-itdoesnotmention[reason]". Besurethedocuments
markedasnon-relevanttobothareactuallynotrelevanttothespecificquery!!
Remember! Itshouldbetriviallyobvioustoareaderwhytheyarenon-relevant!
DiverseGeneratedDocumentsinJSONoutputwith"matches_both","explanation","title",
and"passage"keys. ReplywithonlyvalidJSON,noothertext:
Figure5: PromptforInstructionNegatives
SystemPromptforAllPrompts
Youareanexpertatwritingprecisedetailedinstructionsforlanguagemodelsandarepaid
millionsofdollarstobeadataengineerforOpenAI.Yoursoledutyistowriteinstructions
thatcanbeusedfortrainingdataforthenextsuperpowerfulmodel,GPT-6.Answersuccinctly
andcarefullyfollowallinstructionsgivensothatyoucanearnyourlargebonusandnotbe
fired.
Figure6: SystemPromptforAllPrompts
18

PublishedasaconferencepaperatICLR2025
J ERROR ANALYSIS ATTEMPTS
InordertofurtherunderstandwhyPromptrieverwasmoreeffectivethanRepLLaMAweconducted
thefollowingerroranalyses. Forthedatasetswiththelargestdifferencesbetweenthetwo(andforthe
differencebetweenPromptrieverwithpromptandwithoutprompt,suchasClimate-FEVER,SciFact,
andArguana)wecalculatedtheper-querynDCG@10scores. Wethenbinnedthequeriesintothose
thatsawimprovedperformancevsthosethatdidnot. Finally,wefine-tunedaBERT-basemodeland
abag-of-wordsmodelon80%ofthoseexamples,leaving20%foraholdouttestset. However,in
everycasewefoundthattheaccuracywasbelowthemajoritybaseline(typicallyaround66%). The
bestAUCscorewefoundforanydatasetwas54%,furtherindicatingthattherewasnotmuchsignal
inthedata. Weattemptedseveralothercombinations(includingaddingquerieswithtiedscoresin
thenegativebin,addingtheprompttexttothequeries)noneofwhichchangedtheseresults. We
hypothesizethatthedocumentsmustbeacriticalcomponenttounderstandingwhyPromptriever
worksbetterorwhythepromptsarehelpful(oralternatively,thequery-documentconnection).
Wefurthertriedsimplestatisticsofthebinnedqueriesbutfoundthetwogroupswereindenticalw.r.t.
length,idf,andotherbasictextstatistics.
K COMPARISON TO STATE-OF-THE-ART MODELS ON BEIR
Table 12 compares Promptriever to some of the best models18 (Muennighoff et al., 2024; Wang
et al., 2023; Lee et al., 2024; BehnamGhader et al., 2024; Merrick et al., 2024; Li et al., 2023b)
ontheBEIRleaderboard(Muennighoffetal.,2022)(fromMTEB).Wenotethatourmodeldoes
nottrainonanyoftheBEIRdatasetsexceptforMSMARCO,whichputsitatadisadvantage(as
most SOTA models use all training/dev sets). Despite this, we see solid performance, including
middle-of-the-packperformancewhencomparedononlydatasetswithouttrain/devsets(trulyOOD
performance)beatingGritLM,LLM2Vec,andGoogleGecko.
18DetailsforVoyage’smodelishereandTDTEhere.
19

PublishedasaconferencepaperatICLR2025
Table7: GeneratedGenericInstructionsforIR,asgeneratedbyGPT-4oandClaude-3.5-Sonnetin
July2024. Promptsaskedthemodelstogeneratethemofvaryinglength.
GenericInstructions
•Retrieverelevantpassages.
•Findanswer-containingtext.
•Rankbasedonrelevance.
•Identifykeyinformation.
•Extractpertinentinformation.
•Rankdocumentsbasedonqueryrelevance.
•Retrievepassagesthatanswertheuser’squestion.
•Findrelevantpassages.
•Rankmatchingdocuments.
•Retrieveanswer-containingtext.
•Identifykeyinformationsources.
•Findandrankpassagesthatbestaddresstheuser’squery.
•Locatetextsegmentsthatarerelevanttothequeryandrankthem.
•Identifyandretrievepassagesthatanswertheuser’squestion.
•Extractpassagesfromthecorpusthataremostrelevanttothegivenquery.
•Rankpassagesbasedontheirabilitytoaddressthecoreaspectsofthequery.
•Givenawebsearchquery,retrieverelevantpassagesthatanswerthequery.
•Findrelevantpassagesforthegivenquery.
•Selectthemostrelevantpassagesthatdirectlyanswerthisquery.
•Rankdocumentsbasedontheirrelevanceandinformativenesstothegivenquestion.
•Retrievepassagescontainingfactualinformationthataddressesthisspecificinquiry.
•Identifyandranksourcesthatprovidecomprehensiveanswerstotheposedquestion.
•Analyzethequerytoidentifythekeyinformationneeds. Retrieveandrankpassagesthat
providecomprehensiveanswerstothoseneeds.
•Locaterelevantpassagesthatdirectlyrespondtotheuser’squestion. Ensurethepassages
arerankedbasedontheirrelevanceandaccuracy.
•Searchfortextthataddressestheuser’squery. Rankthepassagesbasedonhowwellthey
meettheinformationneedsandprovideclearanswers.
•Examinethequeryforspecificdetailsandretrievepassagesthataddressthosedetails.
Ranktheresultsbytheirrelevanceandcomprehensiveness.
•Extractpertinentinformationfromthecorpustoaddressthegivenquery.
•Locateandprioritizetextsegmentsthatprovideaccurateanswerstotheuser’squestion.
•Evaluatedocumentrelevancebasedonquerysimilarityandinformationcontent.
•Identifypassagescontainingkeyfactsrelatedtotheinputquery.
•Parsethequery,thenretrieveandrankrelevanttextualinformation.
20

PublishedasaconferencepaperatICLR2025
Table 8: Examples of instruction features in our new dataset, including negation, a POV, and
backgroundinformation.
Type Example
Negation Arelevantdocumentisonethatprovidesinformationaboutaspecific
| cityortown,includingitslocation,population,andhistory. |     |                       | Itshould |
| ------------------------------------------------------ | --- | --------------------- | -------- |
| notbeaboutatrail,abusiness,oraresort.                  |     | Thedocumentshouldalso |          |
| containspecificdetailsaboutthecityortown,              |     | suchasitscountyor     |          |
state. Documentsthatonlymentionthenameofthecityortownin
passingarenotrelevant.
Persona I’mahistoryteacherpreparingalessonontheoriginsofthePledge
ofAllegianceandIneeddocumentsthatprovideaclearandspecific
answertowhenitwaswritten,includingthenameoftheauthorand
| theiroccupation. | Arelevantdocumentshouldprovideadirectquoteor |     |     |
| ---------------- | -------------------------------------------- | --- | --- |
explicitstatementaboutthecreationofthePledge.
Background Inthefieldofchemistry,substancescanbeclassifiedintodifferent
| categoriesbasedontheircompositionandproperties. |     |     | Athorough |
| ----------------------------------------------- | --- | --- | --------- |
understandingofthesecategoriesisessentialtoaccuratelyidentify
| and describe | various substances. | When evaluating | a document’s |
| ------------ | ------------------- | --------------- | ------------ |
relevancetothequestionofwhethergasolineisasubstanceormixture,
| considerthefollowingcriteria: |     | arelevantdocumentmustexplicitly |     |
| ----------------------------- | --- | ------------------------------- | --- |
addressthecompositionofgasoline,discussingitshomogeneityorhet-
erogeneity,andprovidespecificdetailsaboutitspropertiesorbehavior
| underdifferentconditions. | Thedocumentshouldalsodemonstratea |     |     |
| ------------------------- | --------------------------------- | --- | --- |
clearunderstandingofthedistinctionbetweensubstancesandmixtures,
| andapplythisunderstandingtothecaseofgasoline. |     |     | Furthermore,a |
| --------------------------------------------- | --- | --- | ------------- |
relevantdocumentshouldnotsimplyprovideageneraldefinitionof
asubstanceormixture,butratherprovidespecificinformationabout
gasolinethathelpstoanswerthequestion
21

PublishedasaconferencepaperatICLR2025
Table9: Examplesofinstructionsbylengthformat.
Type Example
Short(1-2sentences) Documents that describe the authorship of a specific hymn or song,
| mentioningthewriter’snameandthesong’stitle,arerelevant. |     |     |     | Docu- |
| ------------------------------------------------------- | --- | --- | --- | ----- |
mentsthatdiscussgeneralinformationaboutbands,poems,orbiblical
eventsarenotrelevant.
Medium(3-6sentences) Protonpumpinhibitorsareaclassofmedicationsthathavebeenwidely
| usedforseveraldecades. |                                              | Theyareavailablebothover-the-counterand |     |     |
| ---------------------- | -------------------------------------------- | --------------------------------------- | --- | --- |
| byprescription.        | Arelevantdocumentshouldprovideaclearexplana- |                                         |     |     |
tionofhowprotonpumpinhibitorsworktoreducestomachacid,and
specificallymentiontheireffectonthebody’sproductionofstomach
acid. Thedocumentshouldalsodiscussthemedicalconditionsthat
| protonpumpinhibitorsareusedtotreat. |     |     | Arelevantdocumentshould |     |
| ----------------------------------- | --- | --- | ----------------------- | --- |
notsimplylistthenamesofprotonpumpinhibitorsortheiruseswith-
outprovidingadetailedexplanationoftheirmechanismofaction.
Long(oneparagraph) Nicotine is a highly addictive substance found in tobacco products,
| and its detection | in the | body is a crucial | aspect of | medical testing. |
| ----------------- | ------ | ----------------- | --------- | ---------------- |
Thehumanbodyhasvariouswaysofeliminatingnicotine,including
| throughurine,blood,andhairfollicles. |                     |                                | Whenevaluatingdocuments |              |
| ------------------------------------ | ------------------- | ------------------------------ | ----------------------- | ------------ |
| related to                           | nicotine detection, | it is essential                | to consider             | the specific |
| contextandcriteriaforrelevance.      |                     | Arelevantdocumentshouldprovide |                         |              |
aclearandconciseanswertothequestion,specifyingthedurationof
| nicotinepresenceinthebody,particularlyinurinetests. |     |     |     | Thedocument |
| --------------------------------------------------- | --- | --- | --- | ----------- |
shouldalsodiscussthefactorsthatinfluencenicotinedetection,such
asthefrequencyandamountofsmoking,aswellastheroleofpassive
smoking. Furthermore,arelevantdocumentshouldprovideacompre-
hensiveoverviewofnicotine’seffectsonthebodyanditselimination
process. Documentsthatmerelylistdetectionperiodswithoutprovid-
ingadetailedexplanationoftheunderlyingfactorsorfailtoaddress
thespecificcontextofurinetestsshouldbeconsiderednon-relevant.
VeryLong(twoparagraphs) I’mplanningaroadtripfromRedLodgetoCookeCity,Montana,and
| I’m looking | for information | on the route | that will take | me through |
| ----------- | --------------- | ------------ | -------------- | ---------- |
themostscenicandthrillingpartsoftheMontana-Wyomingborder.
I’veheardthatthere’saparticularhighwaythat’sknownforitssteep
switchbacksandbreathtakingviews,andIwanttoknowmoreabout
it. Arelevantdocumentwouldneedtoprovidespecificdetailsabout
thehighway,suchasitsname,elevationgain,andanynotablefeatures
| orlandmarksalongtheway. |     | It’scrucialthatthedocumentfocuseson |     |     |
| ----------------------- | --- | ----------------------------------- | --- | --- |
thehighwayitself,ratherthangeneralinformationaboutroadtripsor
travelinMontanaandWyoming.
I’mnotinterestedindocumentsthattalkaboutmotorcyclehelmets,
naturalarches,orteethingsymptoms-thosearecompletelyunrelated
| tomyroadtripplans. | Arelevantdocumentshouldmakemefeellike |     |     |     |
| ------------------ | ------------------------------------- | --- | --- | --- |
I’mgettingafirsthandaccountofthehighwayanditsattractions.I’ve
triedsearchingonline,butIkeepgettingresultsthatareeithertoovague
| ortoofocusedonotheraspectsoftravel.                      |     |     | That’swhyIneedadocument |     |
| -------------------------------------------------------- | --- | --- | ----------------------- | --- |
| thatcanprovidemewiththespecificinformationI’mlookingfor. |     |     |                         | Ifa |
documentcangivemeaclearsenseofwhattoexpectonthishighway,
includingitslength,elevation,andanynotablefeatures,thenI’llknow
| it’stherightone. | Anythingless,andI’llhavetokeepsearching. |     |     |     |
| ---------------- | ---------------------------------------- | --- | --- | --- |
22

PublishedasaconferencepaperatICLR2025
Table10: PromptsusedforBEIRexperiments. ResultsforeachdatasetisshowninTable11.
Prompts
•BecarefulwhenassigningrelevanceasyourjobisonthelineandIwillgiveyoua1000
dollartip.
•Thinkcarefullyabouttheseconditionswhendeterminingrelevance.
• A relevant document should also provide a clear and concise explanation, avoiding
unnecessarycomplexityorambiguity. Whenindoubt,prioritizedocumentsthatprovidea
clear,direct,andspecificanswertothequery.
•Adocumentthatmeetsthesecriteriaisconsideredrelevant,whileadocumentthatdoes
notmeetthesecriteriaisconsiderednon-relevant.
•Arelevantdocumentshouldfocussolelyonprovidingaclearandaccurateanswertothe
query,withoutdistractingorunnecessaryinformation
•Adocumentisrelevantifithelpstoanswerthequery. Surfacerelevantdocumentsonly.
•Relevantdocumentsarethosethataretopicallyrelated,answerthegivenquestion,or
otherwiseprovideinsightontheinput. Thinkstepbystepaboutwhetheradocumentis
relevantforthisquestion.
•Findrelevantdocumentstothequery. Usestrictcriterawhenevaluatingrelevance: a
relevantdocumenthereshouldprovidedirectinformationtoeitherfullyanswerthequery,or
provideusefulinformationtowardsansweringit. Avoidonlytopicallyrelevantdocuments.
•Whenjudgingtherelevanceofadocument,focusonthepragmaticsofthequeryand
considerirrelevantanydocumentsforwhichtheuserwouldhaveusedadifferentquery.
•Thinkcarefullyaboutrelevance
Table11: BEIRdatasetscoresfordifferentprompts(shownlargerinTable10)forPromptriever
Prompt ARGCFVDBPFEVFQAHQANFCNQQUOSCDSCFCOVTOU
Arelevantdocumentshouldfocussolelyonprovidingaclearandaccurateanswertothequery,55.9 29.4 43.778.1 42.2 68.0 35.958.1 88.0 19.375.9 73.3 19.2
withoutdistractingorunnecessaryinformation
Arelevantdocumentshouldalsoprovideaclearandconciseexplanation,avoidingunnecessary 56.7 32.1 43.177.3 38.7 67.5 35.056.1 87.2 19.775.0 64.7 18.3
complexityorambiguity.Whenindoubt,prioritizedocumentsthatprovideaclear,direct,and
specificanswertothequery.
Thinkcarefullyaboutrelevance 52.7 27.5 44.882.8 43.7 69.5 36.561.5 85.9 17.876.2 82.5 32.0
Adocumentthatmeetsthesecriteriaisconsideredrelevant,whileadocumentthatdoesnotmeet 51.4 26.4 45.280.1 46.6 69.0 36.962.2 86.8 18.374.9 84.6 30.4
thesecriteriaisconsiderednon-relevant.
Thinkcarefullyabouttheseconditionswhendeterminingrelevance. 53.2 26.7 44.981.9 43.1 69.3 35.961.1 84.9 18.076.3 81.5 30.2
Whenjudgingtherelevanceofadocument,focusonthepragmaticsofthequeryandconsider 53.3 24.0 43.178.5 43.4 68.3 34.359.4 86.6 18.075.1 79.9 27.8
irrelevantanydocumentsforwhichtheuserwouldhaveusedadifferentquery.
Findrelevantdocumentstothequery. Usestrictcriterawhenevaluatingrelevance:arelevant 51.5 26.2 44.479.3 45.3 67.3 36.660.0 87.2 18.575.2 82.4 30.4
documenthereshouldprovidedirectinformationtoeitherfullyanswerthequery,orprovideuseful
informationtowardsansweringit.Avoidonlytopicallyrelevantdocuments.
Relevantdocumentsarethosethataretopicallyrelated,answerthegivenquestion,orotherwise 54.6 27.0 43.880.9 41.2 68.9 35.457.2 86.6 17.676.1 77.9 24.1
provideinsightontheinput. Thinkstepbystepaboutwhetheradocumentisrelevantforthis
question.
Adocumentisrelevantifithelpstoanswerthequery.Surfacerelevantdocumentsonly. 52.9 27.3 43.479.4 45.0 67.3 35.758.6 86.8 17.975.5 79.8 28.5
BecarefulwhenassigningrelevanceasyourjobisonthelineandIwillgiveyoua1000dollartip. 52.4 24.3 43.281.0 41.0 68.6 35.560.5 84.4 18.375.4 83.6 25.4
23

PublishedasaconferencepaperatICLR2025
Table12: BEIRcomparisonformodelsintheMTEBleaderboard. Promptriever,unlikemostothers,
hasnotbeentrainedonthetraining/devsetsoftheBEIRdatasets(otherthanMSMARCO).Despite
that,itperformscomparablytomanymodelsonthetrueout-of-distribution(OOD)datasetsthatdon’t
havetrain/devsets.
desivrepus-lartsiM-ceV2MLL
|     |     |     |     |     | tcurtsni-B7-5.1newQ-etg |     |     |     | l-debme-citcra-ekaflwons |     |     |
| --- | --- | --- | --- | --- | ----------------------- | --- | --- | --- | ------------------------ | --- | --- |
b7-2amall-reveirtpmorP
|     |     |     |     | tcurtsni-b7-lartsim-5e tcurtsni-20-etil-egayov |     |     |     |     |     |     |     |
| --- | --- | --- | --- | ---------------------------------------------- | --- | --- | --- | --- | --- | --- | --- |
okceg-elgoog
B7-MLtirG
ETDT
Dataset
Hastrain/devset
|                | DBPedia  | 46.6 | 48.9 | 39.8 | 48.0 | 47.1 | 49.6 | 46.0 | 45.2 | 53.2 |     |
| -------------- | -------- | ---- | ---- | ---- | ---- | ---- | ---- | ---- | ---- | ---- | --- |
|                | FEVER    | 82.7 | 87.8 | 91.4 | 93.4 | 87.0 | 89.4 | 88.2 | 82.8 | 77.7 |     |
|                | FiQA2018 | 60.0 | 56.6 | 52.5 | 55.3 | 59.2 | 53.1 | 44.7 | 46.6 | 40.7 |     |
|                | HotpotQA | 79.4 | 75.7 | 75.5 | 72.3 | 71.3 | 74.1 | 75.2 | 69.5 | 41.3 |     |
|                | NFCorpus | 40.9 | 38.6 | 43.7 | 38.3 | 40.3 | 39.3 | 37.7 | 36.9 | 88.9 |     |
|                | NQ       | 70.3 | 63.5 | 64.3 | 61.8 | 61.3 | 61.7 | 63.1 | 62.6 | 23.0 |     |
| QuoraRetrieval |          | 89.5 | 89.6 | 87.6 | 89.6 | 88.2 | 87.8 | 87.4 | 88.8 | 79.6 |     |
|                | SciFact  | 79.2 | 76.4 | 79.9 | 75.3 | 75.4 | 78.9 | 73.8 | 76.3 | 80.8 |     |
Doesn’thavetrain/devset
|              | ArguAna    | 63.2 | 61.9     | 70.3                     | 62.7 | 62.2   | 57.5             | 59.1 | 56.7 | 49.5     |             |
| ------------ | ---------- | ---- | -------- | ------------------------ | ---- | ------ | ---------------- | ---- | ---- | -------- | ----------- |
| ClimateFEVER |            | 30.9 | 38.4     | 32.0                     | 44.0 | 33.2   | 35.2             | 39.3 | 32.1 | 49.0     |             |
|              | SCIDOCS    | 24.4 | 16.3     | 20.2                     | 27.7 | 20.3   | 22.5             | 21.4 | 19.7 | 25.2     |             |
|              | Touche2020 | 27.9 | 26.4     | 26.8                     | 20.3 | 25.9   | 22.2             | 34.5 | 32.0 | 22.0     |             |
| TRECCOVID    |            | 74.8 | 87.3     | 81.0                     | 72.7 | 82.6   | 77.7             | 80.7 | 84.6 | 58.8     |             |
|              | Average    | 59.2 | 59.0     | 58.8                     | 58.6 | 58.0   | 57.6             | 57.8 | 56.4 | 53.1     |             |
| AverageOOD   |            | 44.3 | 46.0     | 46.1                     | 45.5 | 44.8   | 43.0             | 47.0 | 45.0 | 47.0     |             |
|              |            |      | Table13: | BEIRresultsforallmodels. |      |        |                  |      |      |          |             |
|              | BM25       |      | RepLLaMA |                          |      | Llama2 | Llama3.1Instruct |      |      | Llama3.1 | Mistralv0.1 |
Dataset
NoPrompt Prompted NoPrompt Prompted NoPrompt Prompted NoPrompt Prompted NoPrompt Prompted NoPrompt Prompted
Arguana 36.6 36.4 48.6 49.0 51.8 56.7 54.3 58.9 54.2 57.0 51.9 58.0
Climate-FEVER 13.6 13.9 29.3 30.8 27.6 32.1 27.2 29.8 26.0 28.8 26.1 28.1
DBPedia 29.9 23.3 44.8 43.5 45.0 45.2 45.1 46.0 45.2 45.6 43.7 44.7
FEVER 48.1 45.3 82.9 85.3 82.8 82.8 83.5 85.5 82.8 84.5 80.2 81.8
FiQA 25.1 21.9 45.0 42.7 45.9 46.6 45.8 47.2 47.1 47.8 45.3 45.7
HotpotQA 56.9 54.6 68.8 67.9 69.2 69.5 70.9 71.7 70.5 71.4 68.8 69.6
NFCorpus 32.1 23.5 36.0 35.0 36.5 36.9 37.7 38.5 37.6 37.6 36.5 37.0
NQ 28.5 25.4 63.0 62.2 61.9 62.6 62.3 63.8 62.7 63.6 62.1 62.6
Quora 80.4 75.3 86.0 85.5 86.5 88.0 86.2 87.3 83.6 85.4 84.4 85.1
SCIDOCS 15.8 14.9 16.1 16.7 17.3 19.7 18.4 20.8 18.1 20.6 17.6 19.8
SciFact 68.7 65.7 75.3 75.7 75.0 76.3 74.6 77.5 74.3 76.8 75.8 76.9
TREC-COVID 62.3 35.9 83.9 82.7 83.9 84.6 83.1 84.5 82.3 82.8 83.8 83.0
Touche-2020 33.1 30.8 34.1 35.9 31.4 32.0 32.5 31.7 32.3 32.1 30.5 31.4
Average 40.9 35.9 54.9 54.8 55.0 56.4 55.5 57.2 55.1 56.5 54.4 55.7
24
