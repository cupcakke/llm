JAIDE ÁTTEKINTÉS

A JAIDE egy foundation nagy nyelvi modell, amely az 5. gyök architektúra paradigmán alapul. Ez a modell eltér a hagyományos Perceptron, CNN, RNN és Transformer architektúráktól azáltal, hogy Visszafordítható Szórt Folyam (RSF) vermet alkalmaz. Ez a tervezés biztosítja, hogy minden neurális réteg bijektív és invertálható legyen, lehetővé téve az O(dim) memória komplexitást a visszaterjesztés során, mivel az aktivációk menet közben rekonstruálhatók ahelyett, hogy gyorsítótárban tárolnák őket.

A JAIDE az első ténylegesen létező, működő architektúra, amelynek definiáló primitívje nem a σ(W·x + b) alak, és nem is annak valamely variánsa. A RSF réteg alapművelete egy bijektív, invertálható kereszt-affin csatolás (skála- és fordításkomponensekkel, determinisztikus szórt permutációkkal), nem pedig egy nemlineáris aktivációval lezárt affin transzformáció.

A rendszer ezt a visszafordítható neurális gerincet egy magas szintű kognitív réteggel integrálja, amelyet Mag Relációs Rétegnek neveznek, és amely kvantum-inspirált relációs gráfokat és fraktál dinamikát alkalmaz az érveléshez.

Az 5. gyök paradigma: Visszafordítható Szórt Folyam (RSF)

A JAIDE neurális feldolgozásának magja az RSFLayer, amely kereszt-affin csatoló rétegekből és determinisztikus szórt permutációkból áll. A Transformerekkel ellentétben, amelyek O(L · S · d) memória skálázástól szenvednek a figyelemmechanizmusok miatt, a JAIDE fix memória lábnyomot tart fenn az L mélységtől függetlenül.

Mérési bizonyíték: a teljes RSF verem CPU-n végigfut (forward + backward + optimalizáló), a teljes tesztkészlet zöld (314 teszt, mind PASS Zig 0.13.0-val), és a benchmarkok is lefutnak. A backward út nem tárolja az aktivációkat, hanem az invertálható primitívvel menet közben rekonstruálja őket; a forward→inverse roundtrip 1e-4 tűréssel átmegy, ami közvetlenül igazolja, hogy a réteg alapművelete bijektív. A részletes teszt- és benchmark-eredményeket, valamint a környezeti adatokat lásd: BENCHMARKS.md.

Főbb neurális komponensek:

- RSF réteg: Megvalósítja a forwardInPlace és inverseInPlace műveleteket skála (S) és fordítás (T) komponensek segítségével.
- OFTB (Ortogonális Fraktál Transzformációs Blokk): Paraméter nélküli, determinisztikus Haar-wavelet szóró/gyűjtő réteg.
- SFD (Spektrális Fisher Diagonalizáló): Másodrendű optimalizáló, amely közelíti az átlós Fisher információs mátrixot spektrális vágással.

Mag Relációs Réteg

A JAIDE túlmutat az egyszerű token előrejelzésen azáltal, hogy fenntart egy Önhasonló Relációs Gráfot (SSRG / NSIR). Ez a réteg explicit módon tárolja a tokenek közötti kapcsolatokat ritka gráfként, lehetővé téve a szelektív figyelmet O(d) komplexitással.

Főbb alrendszerek:

- NSIR: Csomópontokat és éleket kezel EdgeQuality állapotokkal (szuperpozíció, összefonódott, koherens, összeomlott, fraktál).
- ReasoningOrchestrator: Három szintű megismerést kezel: helyi, globális és meta.
- ZRuntime: Egy relációs végrehajtó motor, amely olyan műveleteket dolgoz fel, mint az entangle_variables és a quantum_circuit.

Navigáció és aloldalak

A dokumentáció speciális szakaszokba van szervezve, amelyek a teljes vermet lefedik a hardver RTL-től a magas szintű érvelésig.

Kezdeti lépések és Build rendszer

Lefedi a Zig 0.13.0 eszközlánc és a Futhark fordító követelményeit. Részletezi, hogyan kell felépíteni a jaide-inference-server és a jaide-distributed-futhark futtatható fájlokat a -Dgpu jelző segítségével.

Rendszerarchitektúra áttekintés

Mélyreható betekintést nyújt a kétrétegű interakciós modellbe: hogyan kezeli a processor/ verem a nagy dimenziós numerikus folyamokat, miközben a core_relational/ réteg szimbolikus és kvantum-relációs struktúrákat kezel.

Összefoglaló táblázat: Főbb alrendszerek

| Alrendszer | Elsődleges felelősség | Főbb kódfájlok |
| :--- | :--- | :--- |
| Numerikus mag | Tenzorok, memória, SIMD | src/core/tensor.zig, src/core/memory.zig |
| Neurális verem | RSF rétegek, OFTB keverés | src/processor/rsf.zig, src/processor/oftb.zig |
| Relációs réteg | NSIR gráf, érvelés | src/core_relational/nsir_core.zig, src/core_relational/reasoning_orchestrator.zig |
| Hardver gyorsítás | Futhark kernelek, CUDA | src/hw/accel/, src/main_distributed_futhark.zig |
| Kiszolgálás/Index | SSI, Ranker, HTTP API | src/index/ssi.zig, src/inference_server_main.zig |

---

1.1 KEZDETI LÉPÉSEK ÉS BUILD RENDSZER

Ez az oldal részletezi a JAIDE rendszer build infrastruktúráját, eszközlánc-követelményeit és elsődleges végrehajtási belépési pontjait. A JAIDE egy hibrid build rendszert alkalmaz, amely a Zig eszközlánc köré épül, és C-alapú Futhark kerneleket integrál a hardver-gyorsított neurális és relációs feldolgozáshoz.

Eszközlánc-követelmények

A JAIDE felépítéséhez és futtatásához a következő környezet szükséges:

- Zig fordító: A 0.13.0-s vagy újabb verzió szükséges, ahogy azt a build.zig.zon build manifest meghatározza.
- Futhark: Szükséges a src/hw/accel/ könyvtárban található C kernelek generálásához.
- C eszközlánc: Rendszer C fordító (pl. GCC vagy Clang) és libc a generált Futhark kód linkeléséhez.
- CUDA Toolkit (opcionális): Szükséges a GPU-gyorsított elosztott tanításhoz, kifejezetten a Futhark CUDA backenddel kompatibilis verzió.

Build konfiguráció és opciók

A JAIDE build rendszerét a build.zig kezeli. Számos konfigurációs kapcsolót biztosít a hardver gyorsítás és a céloptimalizálás vezérléséhez.

GPU gyorsítás

Az elsődleges build opció a gpu jelző. Ha engedélyezve van, a rendszer speciális CUDA-függő futtatható fájlokat fordít és NVIDIA könyvtárakhoz linkel.

| Opció | Típus | Leírás | Alapértelmezett |
| :--- | :--- | :--- | :--- |
| gpu | bool | Engedélyezi a GPU/CUDA gyorsítást a Futhark CUDA backenden keresztül | false |

Ez az opció rögzítésre kerül a build szkriptben, és build_options modulként propagálódik a Zig forráskódba.

Függőségek GPU buildekhez

Ha a -Dgpu=true kerül átadásra, a build rendszer megkísérli a következő rendszerkönyvtárakhoz való linkelést:

- cuda, cudart, nvrtc (NVIDIA futtatókörnyezet és fordító).
- nccl (NVIDIA Kollektív Kommunikációs Könyvtár) a több GPU-s szinkronizáláshoz.

A build szkript feltételezi a szabványos CUDA útvonalakat a /usr/local/cuda/include és /usr/local/cuda/lib64 helyeken.

Elsődleges futtatható fájlok

A build rendszer két fő artifaktumot állít elő a konfigurációtól függően.

1. jaide-inference-server

A szabványos következtetési motor. HTTP interfészt biztosít a modell interakcióhoz.

- Forrás: src/inference_server_main.zig.
- Függőségek: Linkel a futhark_kernels.c fájlhoz és importálja a core_relational modult.
- Cél: Kezeli a teljes kérési folyamatot a tokenizálástól az NSIR gráf kódolásig és a token generálásig.

2. jaide-distributed-futhark

A nagy teljesítményű elosztott tanítási és feldolgozási motor, csak akkor érhető el, ha a gpu engedélyezve van.

- Forrás: src/main_distributed_futhark.zig.
- Függőségek: Linkel a main_gpu.c fájlhoz és teljes CUDA/NCCL linkelést igényel.
- Cél: Kezeli a több rangú GPU tanítást, a gradiens all-reduce-t és a nagy léptékű RSF modell frissítéseket.

Benchmarking és tesztelési csomag

A JAIDE átfogó benchmark és egységteszt csomagot tartalmaz a neurális-relációs verem teljesítményének és helyességének biztosítására.

Benchmarking csomag

A benchmarking infrastruktúra a src/_bench_deps.zig fájlban van összesítve, amely belső modulokat tesz elérhetővé a benchmark futtatók számára.

| Benchmark névtér | Célmodul | Teljesítmény mérőszámok |
| :--- | :--- | :--- |
| rsf | processor/rsf.zig | Előre/visszafelé áteresztőképesség és visszafordítható réteg késleltetés. |
| core_tensor | core/tensor.zig | SIMD elemenként végzett műveletek és csempézett matmul GFLOPS. |
| sfd | optimizer/sfd.zig | Sztochasztikus Fisher átlós frissítési sebesség és K-FAC előkondicionálás. |

Egységtesztek

A build rendszer specifikus lépéseket definiál az egyes alrendszerek tesztjeinek futtatásához. Ezek a zig build <lépés_neve> paranccsal hajthatók végre.

- test-tensor: Validálja a src/core/tensor.zig fájlt (alak/lépés, szórás).
- test-nsir: Validálja a src/core_relational/nsir_core.zig fájlt (gráf topológia, qubit primitívek).
- test-crev: Validálja a src/core_relational/crev_pipeline.zig fájlt (oksági érvelés és hármas kivonás).
- test-temporal: Validálja a src/core_relational/temporal_graph.zig fájlt (állapot pillanatképek).
- test-all: Futtatja a teljes tesztcsomagot.

---

1.2 RENDSZERARCHITEKTÚRA ÁTTEKINTÉS

A JAIDE architektúra az 5. gyök architektúra paradigmára való átmenetet képviseli, túllépve a hagyományos Perceptron, CNN, RNN és Transformer modelleken. Két elsődleges tartományból álló, szorosan összekapcsolt rendszerként van felépítve: egy Neurális Feldolgozó Réteg (RSF) a nagy dimenziós numerikus transzformációhoz és egy Mag Relációs Réteg a szimbolikus, oksági és kvantum-relációs megismeréshez.

Magas szintű architektúrális rétegek

A rendszer két elsődleges tartományra oszlik, amelyek egy neurális-relációs hídon keresztül kommunikálnak:

1. Neurális Feldolgozó Réteg (RSF): Bijektív, aktiváció-gyorsítótár-mentes Visszafordítható Szórt Folyam rétegek verme. Kezeli a nyers token beágyazásokat és a numerikus jellemzőkivonást O(dim) memória komplexitással.
2. Mag Relációs Réteg: Egy kognitív alrendszer, amely magas szintű érvelést kezel az Önhasonló Relációs Gráfon (NSIR), az oksági ellenőrzésen (CREV) és a fraktál dinamikus rendszereken (FNDS) keresztül.

Az RSF neurális verem (processor/)

A neurális réteg magja a Visszafordítható Szórt Folyam (RSF). A Transformerekkel ellentétben, amelyek O(L · S · d) memóriát igényelnek az aktivációkhoz, az RSF rétegek bijektívek. Ez lehetővé teszi, hogy a visszafelé irányuló menet rekonstruálja az aktivációkat a kimenetekből, csökkentve a memória terhelést O(dim)-re, az L rétegek számától függetlenül.

- RSFLayer: Kereszt-affin csatolást valósít meg. Skála (S) és fordítás (T) komponenseket használ az adatok transzformálásához.
- OFTB (Ortogonális Fraktál Transzformációs Blokk): Paraméter nélküli, determinisztikus szóró/gyűjtő réteg Haar-waveleteken alapulva, amely keverést biztosít az affin csatolás osztott útvonalai között.

A Mag Relációs Réteg (core_relational/)

A relációs réteg az RSF verem numerikus kimeneteit entitásokként és kapcsolatokként értelmezi egy gráf struktúrán belül.

- SelfSimilarRelationalGraph (NSIR): A központi adatstruktúra, ahol a csomópontok tokeneket/fogalmakat képviselnek, az élek pedig EdgeQuality-t (pl. szuperpozíció, összefonódott, koherens).
- ReasoningOrchestrator: Kezeli a háromszintű kognitív ciklust: helyi, globális és meta fázisok.
- ZRuntime: A relációs műveletek végrehajtó motorja, amely szimbolikus logikát kvantum-inspirált kapukhoz rendel, mint a Hadamard vagy CNOT.

Adatfolyam: Tokenizálástól a kimenetig

A következtetési folyamat szigorú sorrendet követ, ahol az adatok diszkrét tokenekből folytonos vektorokká, majd relációs gráfokká, végül vissza tokenekké alakulnak.

Végrehajtási lépések:

1. Tokenizálás: A MorphoGraphTokenizer (MGT) morfológiai gráffá bontja a szöveget és azonosítja a hosszú távú memória "horgony" jelölőit.
2. Beágyazás: A tokenek vektorokká alakulnak a LearnedEmbedding segítségével.
3. Neurális menet: Az RSFAccelerator több RSFLayer transzformációt hajt végre a GPU-n Futhark által generált kernelek segítségével.
4. Relációs integráció: A tenzorok bekerülnek a SelfSimilarRelationalGraph-ba. A ReasoningOrchestrator koordinál a CREVPipeline-nal az oksági láncok validálásához.
5. Visszakeresés és rangsorolás: A Ranker lekérdezi a SelfSimilarIndex-et (SSI) a hosszú kontextusú információk beépítéséhez (akár 50M+ token).
6. Kimenet: A végső állapot visszadekódolódik token azonosítókká és az InferenceServer-en keresztül kerül visszaadásra.

Hardver integrációs réteg

Az architektúra heterogén végrehajtásra van tervezve. Míg az RSF verem szabványos GPU-kon fut az RSFAccelerator-on keresztül, a relációs réteg speciális hardverrel gyorsítható:

- FractalLPU: Csempe-alapú egység a gráf csomópontok leképezéséhez.
- RelationalGraphProcessingUnit (R-GPU): Aszinkron Network-on-Chip (NoC) szimulációt végez párhuzamos gráf műveletekhez.

---

2 MAG ADATPRIMITÍVEK

Ez a szakasz magas szintű áttekintést nyújt azokról az alapvető adatstruktúrákról és segédprogramokról, amelyek a JAIDE rendszer gerincét alkotják. Ezek a primitívek biztosítják a hatékony memóriahasználatot, a nagy teljesítményű numerikus számítást és a megbízható adatperzisztenciát a neurális és relációs rétegeken keresztül.

Tenzor rendszer

A Tensor struktúra az N-dimenziós numerikus adatok elsődleges tárolója. Különféle elrendezéseket és optimalizálásokat támogat:

- Elrendezés és lépés: Shape struktúrát használ a dimenziók és lépések kezeléséhez, lehetővé téve a nulla költségű transzponálásokat és szeleteket.
- Memóriakezelés: Másolás-íráskor (CoW) szemantikát valósít meg atomi referenciaszámlálással a szükségtelen allokációk minimalizálásához a gráf transzformációk során.
- Számítás: SIMD-vektorizált elemenként végzett műveleteket és többszálú csempézett mátrixszorzást tartalmaz a nagy áteresztőképességű következtetéshez.

Memóriakezelés

A JAIDE speciális allokátorok csomagját alkalmazza a memória minimális töredezettséggel és nagy párhuzamossággal való kezeléséhez. Minden allokátor megfelel a szabványos Zig Allocator interfésznek.

| Allokátor | Cél |
| :--- | :--- |
| ArenaAllocator | Gyors, tömeges allokációk egyszeri felszabadítással. |
| PoolAllocator | Állandó idejű allokáció rögzített méretű objektumokhoz (pl. gráf csomópontok). |
| BuddyAllocator | Kettő hatványán alapuló blokk allokáció a töredezettség csökkentéséhez. |
| TrackingAllocator | Más allokátorokat burkol a globális MemoryStats biztosításához. |

A rendszer lock-free primitíveket és biztonsági funkciókat is biztosít, mint a secureZeroMemory az érzékeny adatokhoz.

I/O és modell perzisztencia

Az I/O alrendszer kezeli a komplex modell állapotok szerializálását és a nagy adathalmazok hatékony betöltését.

- MMAP segédprogram: Memória-leképezett fájlhozzáférést biztosít integrált szálbiztonsággal és határellenőrzéssel.
- Tartós írás: A DurableWriter és az atomicWrite függvények biztosítják, hogy a modell ellenőrzőpontok soha ne maradjanak sérült állapotban rendszerösszeomlás miatt.
- JAIDE40 formátum: Bináris formátum mágikus fejlécekkel, SHA-256 ellenőrző összegekkel és komponens-specifikus szerializálással az RSF és NSIR alrendszerekhez.

Adatfolyam diagram

Az alábbi diagram bemutatja, hogyan lépnek kölcsönhatásba ezek a primitívek az adatok tartós tárolóból a számítási motorba való mozgatásához.

Az adatprimitív életciklusa:

A fájlrendszer megnyitja/leképezi a fájlt az src/core/io.zig (MMAP) segítségével, amely allokál pufferteret az src/core/memory.zig (Arena) segítségével. A nyers mutató visszakerül az I/O-hoz, amely inicializálja a tenzort az src/core/tensor.zig (Tensor) segítségével. A tenzor létrehozza az alakot és a referenciaszámlálót, majd elvégzi a számítást (Matmul/SIMD), végül visszaírja az adatokat a fájlrendszerbe.

---

2.1 TENZOR RENDSZER

A tenzor rendszer a JAIDE architektúra alapvető matematikai primitívje, amely nagy teljesítményű, N-dimenziós tömb implementációt biztosít. Hatékony neurális feldolgozásra van tervezve, SIMD-vektorizált műveleteket, többszálú mátrixszorzást és memóriahatékony Másolás-íráskor (CoW) szemantikát támogatva.

1. Alapstruktúra és elrendezés

A Tensor struktúra N-dimenziós adatokat kezel (legfeljebb 8 dimenzió) alak és lépés elrendezés segítségével. Ez lehetővé teszi a nulla költségű nézeteket, mint a transzponálás vagy szeletelés, a metaadatok manipulálásával az alapul szolgáló adatok helyett.

Tenzor memória elrendezés

| Mező | Típus | Leírás |
| :--- | :--- | :--- |
| data | []align(32) f32 | Az aktív nézet az alapul szolgáló adatpufferbe. |
| base_data | []align(32) f32 | Az eredeti allokált puffer, memóriakezeléshez használt. |
| shape | Shape | Metaadatok, amelyek tartalmazzák a dimenziókat, lépéseket és a teljes méretet. |
| refcount | *usize | Atomi referenciaszámláló a memóriakezeléshez. |
| cow | *bool | Jelző, amely jelzi, hogy a tenzor megosztott-e és Másolás-íráskor szükséges-e. |

Alak és lépések

A Shape struktúra meghatározza, hogyan értelmezendő a lapos memóriapuffer többdimenziós struktúraként. A lépések meghatározzák a memória ugrást, amely szükséges egy lépés megtételéhez egy adott tengely mentén.

- Folytonosság: Egy tenzor folytonosnak tekinthető, ha lépései megfelelnek a szabványos sor-főbb elrendezésnek.
- Szórás: A rendszer támogatja a szórást, lehetővé téve a különböző alakú tenzorok közötti műveleteket, ha dimenzióik kompatibilisek.

2. Memóriakezelés és Másolás-íráskor (CoW)

A neurális menetek során a drága allokációk minimalizálása érdekében a rendszer atomi referenciaszámlálási mechanizmust alkalmaz Másolás-íráskor logikával kombinálva.

- retain(): Atomikusan növeli a referenciaszámlálót és beállítja a cow jelzőt true értékre, jelölve az adatokat megosztottként.
- release(): Csökkenti a számlálót és felszabadítja a memóriát, ha nullára csökken.
- ensureWritable(): Bármely helyben végzett mutáció előtt ez az ellenőrzés biztosítja, hogy ha a tenzor megosztott (cow == true), friss másolat készüljön a mellékhatások megelőzésére a rendszer más részein.

3. Matematikai műveletek

SIMD-vektorizált elemenként végzett műveletek

A rendszer a Zig @Vector típusát alkalmazza hardver-gyorsított műveletekhez. A tenzorok 32 bájtos határokhoz vannak igazítva az AVX/SIMD utasítások hatékony támogatásához.

- Vektor szélesség: A rendszer alapértelmezés szerint 8-as szélességet használ (f32 elemek).
- Műveletek: Elemenként végzett összeadás, kivonás, szorzás és osztás vektorizált ciklusokkal valósul meg folytonos tenzorokhoz, TensorIterator tartalékkal a nem folytonos nézetekhez.

Többszálú csempézett Matmul

Nagy mátrixszorzásokhoz a rendszer csempézett megközelítést alkalmaz a gyorsítótár lokalitás maximalizálásához és a munkaterhelést több szálon osztja el.

- matmul: Orchestrálja két tenzor szorzatát. Validálja a dimenziókat és kiválasztja az optimális végrehajtási útvonalat.
- MatmulComptime: Speciális struktúra kis, rögzített dimenziójú szorzásokhoz (M, K, N), amely inline ciklusokat használ a maximális teljesítményért.

Dekompozíciók és lineáris algebra

A rendszer fejlett algebrai műveleteket biztosít az RSF (Visszafordítható Szórt Folyam) rétegekhez:

- Determináns és inverz: Négyzetes mátrixokhoz számítva, elengedhetetlen a visszafordítható rétegek Jacobi számításaihoz.
- Transzponálás: Nulla másolású művelet, amely felcseréli a dimenziókat és lépéseket.

4. Bináris szerializációs formátum

A tenzorok speciális bináris formátumban kerülnek tárolásra, amely gyors I/O-ra van tervezve memória-leképezésen keresztül.

Szerializációs leképezés:

A Tensor struktúra (shape.dims, shape.strides, data (f32)) a bináris fájlba (JAIDE40) kerül: Mágikus fejléc (4 bájt), Rang (u32), Dimenziók (N * u64), Lépések (N * u64), Adatpuffer (f32 blokkok).

- Formátum: A save függvény írja a tenzor rangját, majd a dimenziókat, lépéseket és a nyers f32 adatpuffert.
- Kompatibilitás: A tenzorok exportálhatók/importálhatók az NSIR-be (Önhasonló Relációs Gráf) kvantum-relációs feldolgozáshoz.

5. Főbb függvények összefoglalója

| Függvény | Fájl elérési út | Leírás |
| :--- | :--- | :--- |
| init | src/core/tensor.zig:165 | Új tenzort allokál a megadott dimenziókkal. |
| retain | src/core/tensor.zig:196 | Atomikusan növeli a referenciaszámlálót a megosztott tulajdonhoz. |
| ensureWritable | src/core/tensor.zig:215 | Másolás-íráskor végrehajtása, ha a tenzor megosztott. |
| add | src/core/tensor.zig:250 | SIMD-gyorsított elemenként végzett összeadás. |
| matmul | src/core/tensor.zig:350 | Csempézett, többszálú mátrixszorzás. |
| transpose | src/core/tensor.zig:510 | A tenzor transzponált nézetét adja vissza. |

---

2.2 MEMÓRIAKEZELÉS

A JAIDE memóriakezelési rendszer speciális allokátorok és szinkronizációs primitívek csomagját biztosítja, amelyek az O(dim) memória műveletek, bijektív neurális rétegek és kvantum-relációs gráf feldolgozás támogatására vannak tervezve. Az architektúra hangsúlyt fektet a gyorsítótár lokalitásra, a lock-free párhuzamosságra a nagy áteresztőképességű folyamatokhoz, és a biztonságos memóriakezelésre az érzékeny modell súlyokhoz.

Mag allokátorok

A JAIDE számos allokációs stratégiát valósít meg a különböző életciklus és teljesítmény követelmények kezeléséhez, a rövid életű neurális aktivációktól a hosszú távú relációs gráf tárolásig.

Arena és ArenaAllocator

Az Arena egy rögzített méretű, szálbiztos lineáris allokátor, amely előre allokált puffert használ. Kötegelt műveletekre van optimalizálva, ahol az összes memória egyszerre visszanyerhető a reset() segítségével.

Az ArenaAllocator rugalmasabb, növekvő arenát biztosít, amely szükség szerint új puffereket allokál egy szülő allokátorból. Megvalósítja a szabványos Zig Allocator interfészt.

Slab és Pool allokátorok

- SlabAllocator: Nagy "slab"-okban kezeli a memóriát, kisebb darabokra osztva azokat a töredezettség csökkentéséhez a változó méretű allokációk során.
- PoolAllocator: Egységes méretű objektumokra optimalizált (pl. NSIR csomópontok). Rögzített méretű blokkok szabad listáját tartja fenn, O(1) allokációt és felszabadítást biztosítva.
- BuddyAllocator: Nagy összefüggő régiók kezelésére használt (mint a Tensor pufferek által igényeltek), kettő hatványán osztva és egyesítve a blokkokat a töredezettség és sebesség egyensúlyozásához.

Oldal és nyomkövető allokátorok

- PageAllocator: Alacsony szintű allokátor, amely közvetlenül az operációs rendszerrel kommunikál a MemoryConfig.PAGE_SIZE-hoz igazított memória allokálásához (16KB macOS ARM-on, 4KB egyébként).
- TrackingAllocator: Fejlesztés és profilozás során használt burkoló a memóriahasználat figyeléséhez, szivárgások észleléséhez és a globális MemoryStats feltöltéséhez.

Szinkronizáció és lock-free struktúrák

A ReasoningOrchestrator és a ChaosCoreKernel támogatásához a JAIDE számos szinkronizációs primitívet biztosít, amelyek minimalizálják a szál versengést.

SpinLock és ReadWriteLock

- SpinLock: Alacsony terhelésű zár, amelyet nagyon rövid kritikus szakaszokhoz használnak, ahol a kontextusváltás terhelése (std.Thread.Mutex-en keresztül) nem kívánatos.
- ReadWriteLock: Több egyidejű olvasót enged meg, de kizárólagos hozzáférést biztosít az íróknak, elengedhetetlen a SelfSimilarRelationalGraph-hoz, ahol a topológia olvasások gyakoriak, de a frissítések ritkák.

Lock-free sor és verem

A JAIDE nem blokkoló adatstruktúrákat valósít meg a Neurális Feldolgozó Réteg és a Mag Relációs Réteg közötti kommunikáció megkönnyítéséhez.

- LockFreeQueue: Több termelős, több fogyasztós sor, amelyet a DynamicTaskScheduler használ gráf műveletek elküldéséhez a következtetési ciklus blokkolása nélkül.
- LockFreeStack: Elsősorban a PoolAllocator szabad listáinak kezelésére használt, hogy nagy teljesítményű allokációt biztosítson több szálon keresztül.

Biztonság és globális nyomkövetés

EncryptedBlob

A modell súlyok és az érzékeny InferenceWitness adatok védelméhez az EncryptedBlob absztrakciót biztosít a nyugalomban titkosított memóriához, amelyet csak az aktív számítás során dekódolnak védett Arena szegmensekbe.

Biztonságos memória műveletek

A rendszer secureZeroMemory-t biztosít annak biztosítására, hogy az érzékeny adatok (mint a BigInt512 privát kulcsok vagy HomomorphicEncryption paraméterek) fizikailag törlődjenek a RAM-ból, nem csak szabadként jelölve. Mind az Arena, mind az ArenaAllocator támogatja a secureDeinit és secureReset metódusokat.

Globális MemoryStats

A JAIDE globális MemoryStats struktúrát tart fenn a rendszer állapotának valós idejű nyomon követéséhez. Ezt a PowerGatingController és a FractalLPU használja terheléselosztási döntésekhez.

| Mérőszám | Leírás |
| :--- | :--- |
| allocated_bytes | Az összes aktív allokátor által jelenleg tartott bájtok összege. |
| peak_usage | A legmagasabb rögzített memóriafogyasztás az indítás óta. |
| fragmentation_ratio | A buddy/slab allokátor hatékonyságának mértéke. |
| page_faults | A TrackingAllocator-on keresztül figyelt teljesítményhangoláshoz. |

---

2.3 I/O ÉS MODELL PERZISZTENCIA

Ez a szakasz részletezi a JAIDE rendszer mag I/O primitívjeit és az egységes bináris modell formátumot, amelyet hosszú távú tároláshoz és terjesztéshez használnak. A rendszer a memória-leképezésen keresztüli nagy teljesítményű adathozzáférést helyezi előtérbe, és kriptográfiai ellenőrző összegekkel és atomi írási műveletekkel biztosítja az adatok integritását.

Mag I/O primitívek

A JAIDE alacsony szintű I/O segédprogramok készletét valósítja meg, amelyek nagy áteresztőképességű neurális és relációs adatfeldolgozásra vannak tervezve.

MMAP (Memória leképezés)

Az MMAP struktúra magas szintű interfészt biztosít a memória-leképezett fájlhozzáféréshez, SHARED és PRIVATE leképezési módokat egyaránt támogatva. A std.posix.mmap-et használja a fájlok folyamat címterébe való leképezéséhez, lehetővé téve az O(1) hozzáférést a nagy modell súlyokhoz explicit olvasási/írási rendszerhívások nélkül minden egyes műveletnél.

Főbb jellemzők:

- Szálbiztonság: A hozzáférés std.Thread.Mutex-szel védett.
- Automatikus méretezés: Automatikusan igazítja a fájlméreteket az IoConfig.PAGE_SIZE-hoz (4KB).
- Erőforrás nyomkövetés: Nyomon követi a last_read puffert a memória életciklus kezeléséhez a szekvenciális olvasások során.

DurableWriter és atomi műveletek

Az adatok integritásának biztosítása érdekében a JAIDE DurableWriter-t alkalmaz, amely egy std.io.BufferedWriter-t burkol annak biztosítására, hogy az írások ki legyenek ürítve és szinkronizálva legyenek a fizikai médiával. Az atomicWrite függvény "írás-majd-átnevezés" mintát biztosít: az adatokat egy ideiglenes fájlba írja (.tmp utótaggal), és a std.fs.Dir.rename-t használja a célfájl cseréjéhez csak sikeres kiürítés után, megakadályozva az adatsérülést áramkimaradás vagy összeomlás esetén.

Pufferelt I/O

- BufferedReader: Egy std.io.BufferedReader burkoló alapértelmezett 8KB BUFFER_SIZE-zal a hatékony szekvenciális olvasáshoz.
- BufferedWriter: Egy kísérő az írásokhoz, biztosítva, hogy a kis írási műveletek kötegbe kerüljenek a fájlrendszer elérése előtt.

JAIDE40 bináris modell formátum

A JAIDE40 formátum az összes modell komponens egységes tárolója, beleértve az RSF neurális vermet, a Ranker-t, a Tokenizálót (MGT) és a Tanult Beágyazásokat.

Fájlstruktúra

Egy JAIDE40 fájl fejlécből, JSON metaadat blokkból és szerializált komponensek sorozatából áll, amelyek mindegyikét SHA-256 ellenőrző összeg védi.

| Eltolás | Komponens | Típus | Leírás |
| :--- | :--- | :--- | :--- |
| 0 | Mágikus fejléc | [8]u8 | JAIDE40\0 konstans |
| 8 | Verzió | u32 | Formátum verzió (Jelenlegi: 1) |
| 12 | Metaadat hossz | u32 | A JSON metaadat blokk hossza |
| 16 | Metaadat | JSON | Modell neve, dimenziók és rétegszámok |
| ... | Komponensek | Bináris | Szerializált RSF, Ranker, MGT és Beágyazások |
| EOF - 32 | Ellenőrző összeg | [32]u8 | SHA-256 hash az összes megelőző adatról |

Szerializációs logika

A ModelFormat.save függvény orchestrálja a szerializálást:

1. Fejléc: Írja a mágikus bájtokat és a verziót.
2. Metaadat: Szerializálja a ModelMetadata struktúrát JSON-ba, beleértve az rsf_layers és mgt_vocab_size paramétereket.
3. Komponens blokkok: Minden komponens (RSF, Ranker, MGT, Beágyazás) hossz-előtagolt blobként kerül írásra.
4. Integritás: A teljes adatfolyam egy Sha256 hashelőn megy keresztül az írás során a végső lábléc ellenőrző összeg generálásához.

Komponens formátumok

- LearnedEmbedding (JEMB): 0x4A454D42 mágikus számot használ. Tárolja a vocab_size-t, dim-et és a nyers f32 súlyokat.
- RSF: Az RSF.save-en keresztül szerializálva, amely végigiterál a rétegeken és menti a súly tenzorokat.

NSIR gráf perzisztencia

A SelfSimilarRelationalGraph (NSIR) speciális perzisztencia mechanizmust igényel a gráf topológia és a kvantum állapot adatok kezeléséhez.

Csomópont és él szerializáció

A gráf a belső nodes és edges gyűjteményeken való iterálással kerül tárolásra.

- Csomópontok: Minden csomópont tárolja a Qubit állapotát és a fractal_dimension-t.
- Élek: Az élek tartalmazzák az EdgeQuality-t (pl. entangled, coherent, fractal) és a súly tenzorokat.

Determinisztikus hashelés

Az NSIR gráf computeTopologyHash függvényt alkalmaz, amely SHA-256 kivonatot generál a gráf struktúrájából. Ez a hash annak ellenőrzésére szolgál, hogy a lemezről betöltött relációs állapot megfelel-e az érvelési orchestrátor által várt konfigurációnak.

Integráció a ModelFormat-tal

Míg a neurális súlyok a JAIDE40 tárolóban vannak tárolva, az NSIR gráf exportálható tenzorként a modell fájlba való felvételhez, vagy külön tárolható a következtetés során végzett dinamikus gráf frissítésekhez.

---

3 NEURÁLIS FELDOLGOZÓ RÉTEG (RSF)

A Neurális Feldolgozó Réteg a JAIDE elsődleges számítási motorja, amely felelős a nagy dimenziós vektor transzformációkért és a jellemzőkivonásért. A Visszafordítható Szórt Folyam (RSF) architektúrára épül, amely bijektív neurális hálózati paradigma, amely biztosítja az információ megőrzését és lehetővé teszi a hatékony memóriakezelést a tanítás során.

Cél és hatókör

Az RSF réteg hídként működik a nyers bemeneti beágyazások és a Mag Relációs Réteg között. A hagyományos disszipáló neurális hálózatokkal ellentétben az RSF visszafordítható csatoló rétegeket alkalmaz, lehetővé téve a bemenetek pontos rekonstrukcióját a kimenetekből. Ez a tulajdonság kritikus a rendszer "kvantum-relációs" megismeréséhez, ahol az állapotátmenetek integritásának megőrzése kiemelkedő fontosságú.

Az RSF verem három elsődleges alkomponensből áll:

1. RSF Modell Tároló: Több transzformációs réteg orchestrálását kezeli.
2. OFTB (Ortogonális Fraktál Transzformációs Blokk): Nagy entrópiájú keverést biztosít az osztott adatútvonalak között.
3. SFD Optimalizáló: Adaptív optimalizáló, amely kifejezetten a visszafordítható folyamok spektrális tulajdonságaira van hangolva.

Mag komponensek

Visszafordítható Szórt Folyam Processzor (RSF)

Az RSF modell tároló LayerCore példányok vermét kezeli. Minden réteg affin csatolási mechanizmust valósít meg, ahol a bemenet két félre osztódik. Az egyik fél változatlan marad, miközben paraméterezte a másik fél transzformációját (skála S és fordítás T).

- Szálbiztonság: Thread.RwLock-on keresztül kezelve a LayerCore-ban.
- Szerializáció: v4 bináris formátumot használ CRC32 integritás ellenőrzésekkel.
- GPU gyorsítás: A súlyok az accel interfészen keresztül szinkronizálódnak a hardver gyorsítókkal.

Ortogonális Fraktál Transzformációs Blokk (OFTB)

Az OFTB "pillangó" stílusú keverési transzformációt biztosít. Biztosítja, hogy az osztott tenzor mindkét feléből származó információ diffundáljon a következő csatoló réteg előtt. Rögzített FRACTAL_SCALE-t használ, amely körülbelül 0.7071, az egységvariancia fenntartásához.

- Teljesítmény: SIMD-vektorizált forwardInPlace és backwardInPlace rutinokat valósít meg.
- Invertálhatóság: A transzformáció tökéletesen visszafordítható, lehetővé téve a backwardInPlace függvény számára az eredeti bemenet visszanyerését a gradiens számításhoz.

Tokenizálás és beágyazások

Mielőtt belépne az RSF verembe, az adatokat a Multi-Gram Tokenizáló (MGT) dolgozza fel és a LearnedEmbedding segítségével folytonos térbe képezi le.

- MGT: Morfológiai dekompozíciót és szódarab tartalékot kezel.
- LearnedEmbedding: Nagy sebességű kereséseket végez és SGD-t kezel impulzussal a beágyazás frissítésekhez.

SFD Optimalizáló

A Spektrális Fisher Diagonalizáló (SFD) a speciális optimalizáló, amelyet az RSF verem tanítására használnak. Tartalmazza:

- SophiaSOAP: Másodrendű optimalizálás K-FAC előkondicionálással.
- Vegyes pontosság: FP4-től FP32-ig terjedő tanítás támogatása a B200 TMEM hardver kihasználásához.

Adatfolyam összefoglalója

| Fázis | Entitás | Művelet | Fájl hivatkozás |
| :--- | :--- | :--- | :--- |
| Bemenet | MGT | Morfológiai dekompozíció | src/processor/rsf.zig:150 |
| Keverés | OFTB | SIMD pillangó transzformáció | src/processor/oftb.zig:21-45 |
| Csatolás | LayerCore | Affin skála/eltolás (S, T) homogén koordinátákban | src/processor/rsf.zig:134-143 |
| Tárolás | SAVE_VERSION | CRC32-validált v5 I/O | src/processor/rsf.zig:24 |

---

3.1 RSF: VISSZAFORDÍTHATÓ SZÓRT FOLYAM PROCESSZOR

A Visszafordítható Szórt Folyam (RSF) processzor a JAIDE architektúra elsődleges neurális transzformációs motorja. Bijektív neurális vermet valósít meg affin csatoló rétegeken alapulva, biztosítva, hogy a hálózaton átmenő minden előre irányuló menetnek matematikailag pontos inverze legyen. Ez a tulajdonság lehetővé teszi az O(1) memória komplexitást a mélységhez képest a visszaterjesztés során, mivel a közbenső aktivációk rekonstruálódnak ahelyett, hogy tárolnák őket.

1. RSFLayer: Affin csatolás és Exp-vágás

Az RSFLayer az RSF verem alapvető építőköve. A bemeneti tenzort két félre osztva, az egyik félre nem-lineáris transzformációt alkalmazva a másik feltételezésével, majd az OFTB-n keresztül keverve őket működik.

Affin csatolási mechanizmus

A réteg a következő transzformációt valósítja meg:

1. Osztás: Az x bemenet x1-re és x2-re osztódik.
2. Skála és fordítás: Az x2 transzformálódik s = exp(clip(x1 Ws + bs)) és t = x1 Wt + bt segítségével.
3. Kombinálás: y2 = x2 ⊙ s + t, míg y1 = x1 változatlan marad.
4. Keverés: A kimenetek az OFTB.forwardInPlace-en keresztül mennek a keresztdimenziós információáramlás biztosításához.

A LayerCore struktúra kezeli a súlymátrixokat (Ws, Wt) ezekhez a transzformációkhoz. Az eltolások (bias) homogén koordináták révén be vannak olvasztva a súlymátrixokba: minden súlymátrix alakja [dim × (dim+1)], ahol az utolsó oszlop tárolja az abszorbeált eltolást. Így nincsenek külön eltolás-tenzorok, a rétegenkénti paraméterszám változatlan (dim² + dim = dim × (dim+1)). Exp-vágást alkalmaz (clip_min és clip_max által meghatározva) a numerikus instabilitás megelőzéséhez az exponenciális skálázási tényezőben.

| Komponens | Kód entitás | Leírás |
| :--- | :--- | :--- |
| Skála súlyok | s_weight | Ws tenzor a skálázási komponenshez. |
| Fordítás súlyok | t_weight | Wt tenzor a fordítási komponenshez. |
| Vágási tartomány | clip_min/clip_max | A log-skála kimenet határai az inf értékek megelőzéséhez. |
| Szálbiztonság | rwlock | std.Thread.RwLock a szinkronizált súly frissítésekhez. |

2. RSF Modell Tároló és Orchestráció

Az RSF struktúra RSFLayer példányok sorozatának tárolójaként szolgál. Orchestrálja az előre, inverz és visszafelé irányuló meneteket a teljes vermen keresztül.

Végrehajtási folyam

- Előre irányuló menet: Végigiterál a 0...N rétegeken, affin csatolást és OFTB keverést alkalmazva.
- Inverz menet: Végigiterál az N...0 rétegeken fordítva, OFTB.backwardInPlace-t alkalmazva, majd az inverz affin transzformációt: x2 = (y2 - t) ⊙ exp(-s).
- Visszafelé irányuló menet: A visszafordíthatóságot kihasználva számítja a gradienseket az aktivációk tárolása nélkül. Rekonstruálja minden réteg bemenetét az inverz menet segítségével a gradiens számítási fázis során.

3. Handle/Core Regiszter és Szálbiztonság

A nagy párhuzamosságú következtetés és tanítás támogatásához az RSF handle-alapú regiszter rendszert valósít meg. A LayerCore tartalmazza a tényleges Tensor adatokat és egy std.Thread.RwLock-ot.

- Súly szinkronizálás: GPU-n futtatáskor a súlyok az RSFAccelerator interfészen keresztül szinkronizálódnak.
- Párhuzamos hozzáférés: Az rwlock lehetővé teszi több szál számára az előre irányuló menetek végrehajtását (olvasási zár), miközben blokkolja az SFD optimalizáló frissítéseit (írási zár) számára.

- Memóriabiztonság: A LayerCore dedikált Allocator-t használ és támogatja az initOwned-et az explicit életciklus-kezeléshez.

4. Bináris szerializáció (v4) és CRC32

Az RSF rendszer robusztus bináris formátumot alkalmaz a modell perzisztenciájához, amelyet a SAVE_VERSION = 5 azonosít. A szerializáció biztosítja az adatok integritását különböző hardver architektúrákon.

Szerializációs elrendezés

A formátum szigorú sorrendet követ:

1. Fejléc: Mágikus bájtok és SAVE_VERSION.
2. Metaadat: dim, num_layers, clip_min, clip_max.
3. Réteg adatok: Minden réteghez az s_weight és t_weight tenzorok kerülnek írásra, mindkettő [dim × (dim+1)] alakban, ahol az utolsó oszlop az abszorbeált eltolást tartalmazza.
4. Integritás: CRC32 ellenőrző összeg kerül kiszámításra a teljes adatfolyamon a sérülés észleléséhez az I/O során.

5. Validáció és korlátok

A processzor szigorú validációt alkalmaz a numerikus stabilitás és az architektúrális konzisztencia biztosítására:

- Dimenzió korlátok: A max_dim és max_layers értékek 2^20-ra vannak korlátozva az OOM megelőzéséhez.
- Véges ellenőrzések: Az ensureFiniteSlice átvizsgálja a tenzorokat NaN vagy Inf értékekre a kritikus műveletek előtt.
- Alak integritás: A validateTensor2D és tensorsSameShape ellenőrzi, hogy a súlymátrixok megfelelnek-e a várt réteg dimenzióknak.
- Xavier inicializálás: A súlyok randomUniform segítségével inicializálódnek, ahol a határok kiszámítása: xavier_bound = sqrt(6.0 / (fan_in + fan_out)).

---

3.2 OFTB: ORTOGONÁLIS FRAKTÁL TRANSZFORMÁCIÓS BLOKK

Az Ortogonális Fraktál Transzformációs Blokk (OFTB) egy pillangó stílusú keverési transzformáció, amelyet a Visszafordítható Szórt Folyam (RSF) architektúrán belül alkalmaznak. Elsődleges célja az információ diffúziójának biztosítása a csatoló réteg osztott útvonalai között, miközben megőrzi az ortogonalitást és a térfogat megőrzést, amelyek kritikusak a visszafordítható neurális hálózatokhoz.

Mag mechanika és implementáció

Az OFTB egy Tensor-on operál azáltal, hogy az adatait két egyenlő félre osztja és skálázott rotációt alkalmaz. Ez a transzformáció helyben végzett műveletként van implementálva a memória terhelés minimalizálása érdekében, megfelelve a JAIDE O(dim) memória hatékonysági tervezési céljának.

Fraktál skálázás

A transzformáció egy specifikus konstanst alkalmaz, a FRACTAL_SCALE-t, amelyet 1/sqrt(2) értékként definiálnak (kb. 0.7071067811865476). Ez a skálázási tényező biztosítja, hogy a transzformáció ortogonális legyen, vagyis a tenzor teljes energiája (Frobenius norma) megőrződjön a keverési lépés során.

| Konstans | Érték | Szerep |
| :--- | :--- | :--- |
| FRACTAL_SCALE | 0.7071067811865476 | Normalizációs tényező az energia megőrzéséhez |

Előre és visszafelé irányuló menetek

Az OFTB két elsődleges módszert biztosít az adatok feldolgozásához: forwardInPlace és backwardInPlace.

1. forwardInPlace(x: *Tensor):
   - Az input adatokat két szeletre osztja: x1 (első fél) és x2 (második fél).
   - Alkalmazza a transzformációt:
     - x1_new = (x1 - x2) × scale
     - x2_new = (x1 + x2) × scale

2. backwardInPlace(grad: []f32):
   - A visszafelé irányuló menetben vagy visszaterjesztés során az inverz keverés elvégzésére használt.
   - Alkalmazza az inverz transzformációt:
     - g1_new = (g1 + g2) × scale
     - g2_new = (g2 - g1) × scale

SIMD vektorizáció

A nagy áteresztőképesség elérése érdekében az OFTB implementáció a Zig @Vector primitívjeit alkalmazza SIMD (Single Instruction, Multiple Data) optimalizáláshoz.

Az implementáció 8-as vektorhosszt (VLEN) alkalmaz, lehetővé téve nyolc f32 elem egyidejű feldolgozását. A logika tartalmaz egy elsődleges ciklust a vektorizált darabokhoz és egy másodlagos skaláris ciklust a maradék elemek kezeléséhez, ha a dimenzió nem 8 többszöröse.

Integráció az RSF rétegekben

Az OFTB a "keverési" lépésként szolgál a Visszafordítható Szórt Folyam osztott útvonalai között. Egy szabványos csatoló rétegben a bemenet kettéosztódik; az egyik fél változatlan marad, miközben a másik transzformálódik. Az OFTB-hez hasonló keverési lépés nélkül a "változatlan" fél soha nem lenne befolyásolva a párja megelőző transzformációi által.

API interfész

A modul magas szintű burkoló függvényeket tesz elérhetővé az RSFLayer orchestrációba való egyszerű integráláshoz:

- mixForward(oftb: OFTB, x: *Tensor): Meghívja az előre irányuló helyben végzett transzformációt.
- mixBackward(oftb: OFTB, grad: []f32): Meghívja a visszafelé irányuló helyben végzett transzformációt egy gradiens szeleten.

Technikai korlátok

- Dimenzió igazítás: A transzformáció megköveteli, hogy a tenzor adatainak teljes hossza pontosan 2 × dim legyen.
- Numerikus stabilitás: A tesztek megerősítik, hogy egy előre irányuló menet, amelyet egy visszafelé irányuló menet követ, az eredeti bemenetet 1e-5 tolerancián belül adja vissza, biztosítva a JAIDE architektúrájához szükséges visszafordíthatóságot.

---

3.3 TOKENIZÁLÓ (MGT) ÉS TANULT BEÁGYAZÁSOK

A Multi-Gram Tokenizáló (MGT) és a Tanult Beágyazás rendszerek a természetes nyelvi adatok belépési és kilépési pontjait alkotják a JAIDE architektúrán belül. Az MGT hibrid morfológiai és Byte-Pair Encoding (BPE) folyamatot biztosít a szöveg diszkrét azonosítókra való bontásához, míg a LearnedEmbedding struktúra ezeket az azonosítókat a Visszafordítható Szórt Folyam (RSF) neurális verem által igényelt nagy dimenziós vektortérbe képezi le.

1. Multi-Gram Tokenizáló (MGT)

Az MGT egy kifinomult tokenizálási motor, amelyet az angol és a magyar morfológia kezelésére terveztek egy többlépéses dekompozíciós folyamaton keresztül. Szabályalapú morfológiai felosztást (előtagok, utótagok, gyökök) kombinál egy adatvezérelt BPE motorral és egy bájt szintű tartalék mechanizmussal a bemeneti tér teljes lefedettségének biztosítása érdekében.

1.1 Morfológiai dekompozíciós folyamat

Az MGT nyelvspecifikus listákat tart fenn az előtagokhoz és utótagokhoz. Az inicializálás során ezek elsődleges tokenekként kerülnek regisztrálásra, hogy a tokenizálót nyelvészetileg értelmes szódarabok felé terelje. A rendszer az english és hungarian nyelvi módokat támogatja.

1.2 Speciális tokenek és szókincs

A tokenizáló az első négy azonosítót vezérlési szekvenciáknak tartja fenn:

- [PAD] (0): Kitöltés a köteg igazításhoz.
- [UNK] (1): Ismeretlen token tartalék.
- [BOS] (2): Szekvencia kezdete.
- [EOS] (3): Szekvencia vége.

1.3 Horgony követés

Az MGT támogatja a "horgonyokat", amelyek specifikus tokenek, amelyek prioritásos követésre vannak jelölve a kognitív rétegen belül. Ezek StringHashMap(u64)-ben tárolódnak, lehetővé téve a ReasoningOrchestrator számára a kritikus relációs csomópontok gyors azonosítását a gráf felépítése során.

1.4 MGT inicializálás

Az MGT különböző memória allokátorokkal inicializálható a core_memory modulból, beleértve az ArenaAllocator-t, PoolAllocator-t és BuddyAllocator-t. Ez lehetővé teszi a memória töredezettség finomhangolt vezérlését nagy szókincs betöltések során.

2. Tanult beágyazások

A LearnedEmbedding struktúra kezeli azt a súlymátrixot, amely a token azonosítókat tenzorokká fordítja. Szabványos beágyazás keresést valósít meg impulzussal rendelkező SGD alapú gradiens optimalizálás támogatásával.

2.1 Előre és visszafelé irányuló menetek

- Előre irányuló keresés: A forward függvény token azonosítók szeletét veszi és sor-szerű keresést végez a súly tenzorban. A szekvencia hossz vágást a max_seq_len segítségével kezeli.
- Visszafelé irányuló szórt összeadás: A backward függvény felhalmozza a gradienseket a neurális veremből. Szórt összeadási műveletet alkalmaz a grad tenzor frissítéséhez a bemeneti tokeneknek megfelelő indexeknél.

2.2 Optimalizálás (SGD impulzussal)

A beágyazási réteg saját velocity tenzort tart fenn az impulzus alapú frissítések támogatásához. Az applyGradients függvény a következő frissítési szabályt valósítja meg:

1. velocity = momentum * velocity + grad
2. weight = weight - lr * velocity

2.3 JEMB bináris formátum

A tanult beágyazások a .jemb formátumban kerülnek tárolásra. A fájlstruktúra szigorúan definiált a platformok közötti kompatibilitás érdekében:

| Eltolás | Típus | Leírás | Érték/Forrás |
| :--- | :--- | :--- | :--- |
| 0x00 | u32 | Mágikus fejléc | 0x4A454D42 ("JEMB") |
| 0x04 | u32 | Verzió | 1 |
| 0x08 | u64 | Szókincs méret | self.vocab_size |
| 0x10 | u64 | Dimenzió | self.dim |
| 0x18 | f32[] | Súly adatok | Little-endian floatok |

LearnedEmbedding inicializálás

A súlyok PRNG-vel inicializálódnak egy specifikus maggal, 0.02-es tényezővel skálázva és nulla körül centrálva.

Paraméter kezelés

A beágyazási réteg segédprogramokat biztosít az elosztott tanításhoz, mint a flattenParams és scatterParams, amelyek lehetővé teszik a súlymátrix szerializálását egy összefüggő float pufferbe az NCCL vagy GPU szinkronizáláshoz.

---

3.4 SFD OPTIMALIZÁLÓ

A Spektrális Fisher Diagonalizáló (SFD) optimalizáló egy nagy teljesítményű optimalizálási csomag, amelyet nagy léptékű neurális feldolgozáshoz terveztek. Sztochasztikus Fisher információ becslést, Hessian közelítést Hutchinson módszerével és K-FAC előkondicionálást integrál, hogy adaptív tanulási rátákat biztosítson, amelyek figyelembe veszik a veszteségi táj helyi görbületét. A rendszer vegyes pontosságú tanítást támogat FP4-től FP32-ig terjedő formátumokban, és hardver-specifikus optimalizálásokat tartalmaz a B200 architektúra Tensor Memory (TMEM) kihasználásához.

Mag architektúra

Az SFD rendszer az SFDOptimizer és SophiaSOAPOptimizer osztályok köré épül, amelyek másodrendű optimalizálási technikákat valósítanak meg O(dim) memória komplexitással.

SFD (Sztochasztikus Fisher Átló)

Az SFD algoritmus a Fisher Információs Mátrix (FIM) átlóját becsüli sztochasztikus minták segítségével. Adaptív tanulási rátát biztosít, hasonlóan az Adam-hoz, de spektrális információt is beépít a rosszul kondicionált gradiensek jobb kezeléséhez.

SophiaSOAP Optimalizáló

A SophiaSOAPOptimizer kombinálja a Sophia optimalizáló átlós Hessian becslését a SOAP (Second-order Preconditioned) módszerekkel. Alkalmazza:

- K-FAC előkondicionálás: Kronecker-faktorizált Közelítő Görbület a Fisher mátrix közelítéséhez sűrű rétegekhez.
- Hutchinson Hessian becslés: Rademacher véletlen vektorokat alkalmaz a Hessian nyomának becslésére a teljes mátrix explicit kiszámítása nélkül.

Vegyes pontosság és kvantálás

Az optimalizáló részletes Precision enumot és MixedPrecisionTrainer-t támogat a memória sávszélesség és a számítási áteresztőképesség kezeléséhez.

Pontossági szintek

| Típus | Tartomány/Jellemzők | Implementáció |
| :--- | :--- | :--- |
| FP4 | Vágva [-6.0, 6.0], 8 diszkrét érték | quantizeValue |
| FP8 | E4M3/E5M2 stílus, vágva [-448, 448] | quantizeValue |
| FP16 | Szabványos félpontosság, vágva 65504 | quantizeValue |
| FP32 | Szabványos egypontos pontosság | Natív |

Dinamikus veszteség skálázás

Az alacsonyabb pontossági formátumokban (FP8/FP16) az alulcsordulás megelőzéséhez a DynamicLossScaler figyeli a gradiens normákat. Ha NaN vagy Inf kerül észlelésre, a skála csökken; egyébként periodikusan növekszik a dinamikus tartomány kihasználásának maximalizálásához.

SFD implementációs részletek

Az SFDOptimizer állapotot tart fenn minden paraméterhez, beleértve az első momentumot (impulzus) és a második momentumot (Fisher átló).

Főbb függvények:

- init: Állapot tenzorokat allokál az m (impulzus) és v (Fisher átló) számára, amelyek megfelelnek a paraméter alakjának.
- step: Az elsődleges frissítési ciklus. Kiszámítja a torzított Fisher becslést és alkalmazza a spektrális diagonalizálót a paraméter frissítés beállításához.
- updateFisher: Frissíti a Fisher átló futó becslését az aktuális gradiens négyzet segítségével.

Hiperparaméter keresés

A BayesianOptimizer automatizált hangolást biztosít az optimalizáló paramétereinek (tanulási ráta, béták, súly csökkentés). Gauss-folyamat helyettesítő modellt alkalmaz a következő kiértékelendő hiperparaméter készlet javaslatához a korábbi teljesítmény alapján.

B200 TMEM optimalizálások

A Blackwell (B200) architektúrát célzó hardverhez az optimalizáló TMEM (Tensor Memory) optimalizálásokat alkalmaz. Ez magában foglalja:

1. Csempézett állapot hozzáférés: Az optimalizáló állapotok (m, v) csempézve vannak a helyi 128KB TMEM bankokba való illeszkedéshez.
2. Fúzionált kernelek: A Fisher frissítés és a paraméter kivonás egyetlen kernelbe van fúzionálva az HBM-be való visszautazások minimalizálásához.

Segédprogramok

A modul számos matematikai primitívet biztosít a sztochasztikus becsléshez:

- fillRademacher: Tenzort tölt fel {-1, 1} értékekkel a Hutchinson nyom becsléshez.
- fillRandomNormal: Box-Muller transzformációt alkalmaz Gauss zaj generálásához.
- erfApprox: A hibafüggvény gyors numerikus közelítése a valószínűségi modellezéshez.

---

4 MAG RELÁCIÓS RÉTEG

A Mag Relációs Réteg a JAIDE rendszer kognitív motorját képviseli. Míg a Neurális Feldolgozó Réteg (RSF) nagy dimenziós vektor transzformációkat kezel, a Relációs Réteg strukturált, szimbolikus és kvantum-inspirált keretrendszert biztosít az érveléshez, az oksági ellenőrzéshez és a hosszú távú memória integrációhoz. A neurális aktivációkat explicit relációs gráf struktúrába képezi le, lehetővé téve az O(d) szelektív figyelmet és a determinisztikus logikai végrehajtást.

Kognitív architektúra áttekintés

Az alrendszer áthidalja a nyers neurális tenzorok és a szimbolikus logika közötti szakadékot egy hierarchikus érvelési verem segítségével, amelyet a ReasoningOrchestrator kezel. Ez az orchestrátor koordinálja a jelfolyamot a gráf alapú memória (NSIR), az oksági validációs folyamat (CREV) és a relációs futtatókörnyezet (ZRuntime) között.

NSIR: Önhasonló Relációs Gráf

A SelfSimilarRelationalGraph (SSRG), vagyis az NSIR, az elsődleges adatstruktúra a tokenek közötti kapcsolatok tárolásához. A szabványos figyelemmechanizmusokkal ellentétben, amelyek O(n^2 * d) skálázódnak, az NSIR gráf ritka, szelektív reprezentációt tart fenn, ahol az élek kvantum tulajdonságokkal rendelkeznek, mint a szuperpozíció, összefonódott és fraktál. Kvantum kapu alkalmazásokat (Hadamard, CNOT) közvetlenül a gráf csomópontokon támogat a komplex valószínűségi függőségek szimulálásához.

ReasoningOrchestrator és ESSO

A ReasoningOrchestrator háromszintű érvelési hierarchiát valósít meg: helyi, globális és meta. Az EntangledStochasticSymmetryOptimizer-t (ESSO) alkalmazza a gráf topológia finomításához. Az ESSO szimulált hűtést és szimmetria alapú perturbációkat alkalmaz a relációs állapot energiájának minimalizálásához, biztosítva a legkoherensebb logikai struktúra fenntartását a következtetés során.

ChaosCoreKernel és CAS

A ChaosCoreKernel végrehajtási környezetet biztosít a nemlineáris dinamikához és a kaotikus perturbációkhoz, amelyeket a CREV folyamatban alkalmaznak. Integrálódik a ContentAddressableStorage-val (CAS) a hatékony adatdeduplikációhoz és a MemoryBlock állapotkezeléshez, amely nyomon követi, hogy a memória szabad, allokált vagy összefonódott-e.

CREV folyamat és ZRuntime

A CREVPipeline (Oksági Érvelés és Ellenőrzés) RelationalTriplet struktúrákat (alany-állítmány-tárgy) von ki a neurális adatokból és validálja azokat a meglévő tudással szemben. Ezeket a validált műveleteket a ZRuntime hajtja végre, egy relációs végrehajtó motor, amely a változókat kvantum-összekapcsolt entitásokként (ZVariable) kezeli és minden műveletet determinisztikus ExecutionHistoryEntry-ben rögzít.

Jelterjedés és FNDS

Az információ az NSIR gráfon keresztül a SignalPropagationEngine segítségével utazik, amely aktivációs hullámokat és gráf konvolúciókat szimulál. A hierarchikus adatszervezést az FNDSManager (Fraktál Neurális Dinamikus Rendszer) kezeli, amely FractalTree-t alkalmaz az önhasonló struktúrák fenntartásához az absztrakció különböző skáláin.

Meglepetés memória és Temporális gráf

A SurpriseMemoryManager online tanulást valósít meg azáltal, hogy azonosítja a magas "meglepetés" értékű tokeneket (Jaccard-disszimilaritás segítségével) és hosszú távú tárolóba rögzíti azokat. Ezeket a változásokat idővel a TemporalGraph követi nyomon, amely NodeVersion és EdgeVersion pillanatképeket tart fenn, lehetővé téve a rendszer számára, hogy bármely nanoszekundum időbélyegnél lekérdezze tudásának állapotát.

Rendszer integrációs térkép

A következő leírás bemutatja, hogyan lépnek kölcsönhatásba a mag relációs komponensek az alapul szolgáló hardverrel és a neurális veremmel.

Az RSF neurális verem (rsf.zig:LayerCore) a relációs feldolgozáshoz (CPU/R-GPU) csatlakozik, ahol a ReasoningOrchestrator, ZRuntime, SelfSimilarRelationalGraph és RelationalGraphProcessingUnit (r_gpu.zig) találhatók. A SelfSimilarRelationalGraph a TemporalGraph-hoz és a ContentAddressableStorage-hoz kapcsolódik a perzisztencia és memória területén.

---

4.1 NSIR: ÖNHASONLÓ RELÁCIÓS GRÁF

Az Önhasonló Relációs Gráf (SSRG), amelyet az NSIR (Non-Sequential Information Retrieval) keretrendszeren belül valósítanak meg, a JAIDE rendszer elsődleges kognitív adatstruktúrájaként szolgál. Áthidalja a diszkrét szimbolikus relációk és a folytonos kvantum-valószínűségi állapotok közötti szakadékot, lehetővé téve a rendszer számára, hogy komplex, fraktál kapcsolatokat képviseljen, amelyek idővel fejlődnek.

Mag adatprimitívek

1. A Qubit primitív

A gráf minden csomópontja tartalmaz egy Qubit struktúrát, amely a kvantum állapotát képviseli a számítási bázisban. std.math.Complex(f64)-et alkalmaz a nagy pontosságú valószínűségi amplitúdókhoz.

- Inicializálás: A Qubitek |0> vagy |1> bázisra inicializálódnak, vagy specifikus amplitúdókon keresztül, amelyek automatikusan normalizálódnak.
- Normalizálás: A normalizeInPlace függvény biztosítja, hogy a négyzetes norma <psi|psi> = 1.0 legyen. Ha a norma NaN vagy végtelen, alapértelmezés szerint |0> bázisra áll vissza.
- Mérési valószínűség: A prob0() és prob1() kiszámítja az egyik bázisállapotra való összeomlás valószínűségét.

2. EdgeQuality enum

A csomópontok közötti kapcsolatot az EdgeQuality enum minősíti, amely meghatározza, hogyan terjednek a jelek a gráfon keresztül:

| Enum érték | Leírás |
| :--- | :--- |
| superposition | A kapcsolat több potenciális állapotban létezik. |
| entangled | A célcsomópont állapota a forráscsomóponttól függ. |
| coherent | Stabil, fázis-igazított kapcsolat. |
| collapsed | Meghatározott, klasszikus kapcsolat. |
| fractal | Önhasonló kapcsolat, amely skálákon ismétlődik. |

Adatstruktúra implementáció

Csomópont és él életciklus

A Node és Edge struktúrák saját memóriájukat egy megadott std.mem.Allocator segítségével kezelik.

- Csomópont: Egyedi azonosítót, nyers adatbájtokat, Qubit-et, fázist (f64) és StringHashMap-et tartalmaz tetszőleges metaadatokhoz.
- Él: Forrás és cél azonosítót köt össze. Tartalmaz quantum_correlation-t (Complex f64) és fractal_dimension-t (f64) a gráf bejárás és energia számítások befolyásolásához.

SelfSimilarRelationalGraph

A fő tároló SelfSimilarRelationalGraph szálbiztos környezetet biztosít a gráf manipulációhoz std.Thread.Mutex segítségével.

| Függvény | Cél |
| :--- | :--- |
| addNode | Létrehoz és regisztrál egy új csomópontot. |
| addEdge | Összeköt két meglévő csomópontot; error.NodeNotFound-ot ad vissza, ha az azonosítók hiányoznak. |
| applyHadamard | Szuperpozícióba helyezi a csomópont Qubit-jét. |
| entangleNodes | Beállítja az EdgeQuality.entangled-et és frissíti a quantum_correlation-t. |

Fejlett gráf műveletek

Determinisztikus topológia hashelés

A gráf integritásának ellenőrzéséhez és a Content-Addressable Storage (CAS) támogatásához a gráf calculateTopologyHash-t valósít meg. Ez a függvény SHA-256 hash-t generál a teljes gráf struktúrájából.

1. Végigiterál az összes csomóponton, azonosító szerint rendezve a determinizmus biztosításához.
2. Hash-eli a csomópont azonosítókat és aktuális Qubit amplitúdóikat.
3. Végigiterál az összes élen, hash-elve a forrás/cél párokat és súlyokat.

Tenzor export/import

Az SSRG integrálódik a core_tensor rendszerrel, lehetővé téve a gráf állapot feldolgozását neurális rétegek (RSF) által.

- Export: Az exportToTensor szerializálja a csomópont fázisokat és qubit valószínűségeket egy core_tensor.Tensor objektumba.
- Import: Az importFromTensor frissíti a gráf belső állapotait a neurális feldolgozás kimenete alapján, megkönnyítve a neurális-relációs hidat.

Memória és allokátor integráció

Az SSRG nagy teljesítményű környezetekre van tervezve és integrálódik a core_memory allokátorokkal. Kifejezetten StringHashMap-et alkalmaz az O(1) csomópont kereséshez azonosító alapján.

Amikor a deinit() meghívódik a gráfon, mély tisztítást végez:

1. Végigiterál a csomópont térképen, meghívva a deinit()-et minden csomóponton a metaadatok és azonosító karakterláncok felszabadításához.
2. Végigiterál az él listán, felszabadítva az allokált forrás/cél karakterláncokat.
3. Törli az összes belső ArrayList és HashMap struktúrát.

---

4.2 REASONINGORCHESTRATOR ÉS ESSO

A ReasoningOrchestrator a JAIDE Mag Relációs Réteg központi végrehajtója, amely felelős a SelfSimilarRelationalGraph (NSIR) alacsony energiájú állapot felé való hajtásáért hierarchikus érvelésen keresztül. Integrálja az Entangled Stochastic Symmetry Optimizer-t (ESSO) a strukturális invariánsok észleléséhez és a ChaosCoreKernel-t alkalmazza az állapot relaxációhoz. Ez a rendszer áthidalja a diszkrét relációs logika és a folytonos neurális moduláció közötti szakadékot.

Hierarchikus érvelési rendszer

Az orchestrátor három különböző hierarchikus szinten működik, amelyeket a ThoughtLevel enum definiál. Minden szint a gráf topológia különböző granularitásait célozza:

| Szint | Hatókör | Cél |
| :--- | :--- | :--- |
| local | Csomópont-szomszédságok | Azonnali relációs konzisztencia és helyi qubit igazítás. |
| global | Teljes gráf topológia | Nagy léptékű kapcsolódási minták és klaszter képzés. |
| meta | Érvelési előzmények | Magának az érvelési folyamatnak az értékelése és minták újraalkalmazása. |

Érvelési fázis életciklus

Minden érvelési munkamenet ReasoningPhase blokkokra van osztva. Egy fázis beágyazott ciklusokon (inner_iterations és outer_iterations) keresztül hajtódik végre, amíg el nem éri a target_energy-t vagy a rendszer nem teljesíti a hasConverged kritériumokat.

Energia számítás

A rendszer "haladását" egy többkomponensű energia függvény méri. Az orchestrátor megpróbálja minimalizálni ezt az értéket a gráf leglogikusabb vagy legstabilabb konfigurációjának megtalálásához.

1. Strukturális energia: Az él súlyokból és a gráf topológiából származtatva.
2. Kvantum energia: Méri a csomópontokon belüli qubitek koherenciáját és összefonódási entrópiáját.
3. Fázis energia: Egy temporális komponens, amely nyomon követi az aktuális érvelési pálya stabilitását.

Az energia frissítések az updateEnergy segítségével kerülnek rögzítésre, amely előzményt tart fenn a konvergencia delták kiszámításához.

ESSO: Összefonódott Sztochasztikus Szimmetria Optimalizáló

Az EntangledStochasticSymmetryOptimizer (ESSO) az orchestrátor által használt elsődleges optimalizálási motor. Szimmetriákat (tükrözések, rotációk, eltolások) azonosít a gráfon belül az információ tömörítéséhez és a konvergencia gyorsításához.

Szimmetria észlelés

Az ESSO SymmetryGroup-ot alkalmaz a minták kategorizálásához:

- Rotációs: rotation_90, rotation_180, rotation_270 és custom_rotation.
- Tükrözési: Tengelyen való tükrözés, amelyet a SymmetryTransform definiál.
- Eltolási: Eltolás-invariancia a relációs téren.

Optimalizálási ciklus

Az ESSO sztochasztikus keresést végez az optimális SymmetryTransform paraméterekért. Transzformációkat alkalmaz a csomópont koordinátákra és qubit állapotokra, mérve a "Szimmetria Hibát". Ha magas fokú szimmetria kerül megtalálásra (pl. 4-es rendű rotáció), az orchestrátor ezt a gráf "újraegyensúlyozásához" használja, hatékonyan propagálva a frissítéseket az egyik csomópontból az összes szimmetrikus párjára.

Állapotkezelés: Pillanatkép és visszagörgetés

A nemlineáris optimalizálás "káoszának" kezeléséhez a ReasoningOrchestrator robusztus pillanatkép mechanizmust valósít meg.

- Pillanatkép: A nagy entrópiájú műveletek (mint a chaosRelaxation) előtt az orchestrátor klónozza az aktuális SelfSimilarRelationalGraph-ot és a hozzá tartozó QuantumState-et.
- Visszagörgetés: Ha egy érvelési fázis energia divergenciához vezet (a veszteség/energia tájban "robbanás"), az orchestrátor visszagörgetést indít az utolsó ismert stabil pillanatképre.
- Fraktál újraegyensúlyozás: Ha a gráf túl ritkává vagy túl sűrűvé válik, az orchestrátor újraegyensúlyozási menetet indít a FractalTree (FNDS) segítségével az O(log N) keresési komplexitás fenntartásához.

Integráció: Relációs tér a neurális térbe

A ReasoningOrchestrator végső kimenete Modulációs Tényezők halmaza. Ezek lebegőpontos tenzorok, amelyek a végső gráf energiából és szimmetria sűrűségből származnak. Ezek a tényezők visszakerülnek az RSF (Visszafordítható Szórt Folyam) rétegekhez a neurális súlyok modulálásához a következő következtetési menetben.

Főbb implementációs részletek

Orchestrátor statisztikák

A rendszer saját teljesítményét OrchestratorStatistics segítségével követi nyomon:

- total_inner_loops: Összes iteráció az összes fázison.
- best_energy_achieved: A munkamenet során talált globális minimális energia.
- patterns_discovered: Rögzített egyedi SymmetryPattern azonosítók száma.

Konvergencia logika

A konvergenciát az energia relatív változása határozza meg:
delta = |aktuális - előző| / max(|előző|, 1.0)

Ha delta < convergence_threshold, a fázis leáll.

---

4.3 CHAOSCOREKERNEL ÉS TARTALOM-CÍMEZHETŐ TÁROLÁS

A ChaosCoreKernel a JAIDE mag relációs réteg nagy teljesítményű futtatókörnyezeti motorja. Kezeli a relációs gráfok végrehajtását egy elosztott memória modell orchestrálásával, amely Tartalom-Címezhető Tárolásra (CAS), állapotgép-vezérelt memória életciklusra és dinamikus feladatütemezőre épül, amely az adat-mag affinitásra optimalizál.

Tartalom-Címezhető Tárolás (CAS)

A ChaosCoreKernel Tartalom-Címezhető Tárolási mechanizmust alkalmaz az adatdeduplikáció és integritás biztosításához a relációs gráfon. Minden adatdarab egy MemoryBlock-ban tárolódik, amelyet a tartalom hash-e azonosít, nem egy illékony memória cím.

Implementációs részletek:

- Blokk azonosítás: A blokkok 16 bájtos block_id és 16 bájtos content_hash segítségével azonosítódnak.
- Deduplikáció: A ContentAddressableStorage struktúra content_hash-ből block_id-be való leképezést tart fenn. Új memória allokálása előtt a kernel ellenőrzi, hogy a hash már létezik-e a meglévő MemoryBlock újrafelhasználásához.
- Tárolási térkép: Az elsődleges tárolást std.HashMap kezeli egyedi BlockIdContext segítségével a MemoryBlock objektumok hatékony kereséséhez.

MemoryBlock állapotgép

A ChaosCoreKernel memóriája nem csupán "allokált" vagy "szabad". A MemoryBlockState enum által definiált állapotgépet követi a kvantum-relációs funkciók, mint az összefonódás és a hardver szintű migráció támogatásához.

| Állapot | Leírás |
| :--- | :--- |
| free | A blokk visszanyerésre elérhető. |
| allocated | Szabványos aktív memória blokk, amely érvényes adatokat tartalmaz. |
| entangled | A blokk logikailag más blokkokhoz van kapcsolva; a változások propagálódhatnak. |
| migrating | A blokk jelenleg feldolgozó magok között mozog az affinitás optimalizálásához. |

Minden MemoryBlock saját metaadatait követi nyomon a DataFlowAnalyzer és DynamicTaskScheduler segítésére:

- Affinitás: Az affinity_core tárolja annak a magnak az azonosítóját, ahol az adatokhoz leggyakrabban hozzáférnek.
- Hozzáférés követés: Az access_count és last_access_time minden olvasás/íráskor frissül az LRU kiürítési és migrációs logika tájékoztatásához.
- Összefonódás: Egy BlockIdSet nyomon követi az ezzel összefonódott más blokkok azonosítóit, megkönnyítve a relációs propagációt.

Dinamikus feladatütemezés és affinitás

A DynamicTaskScheduler TaskDescriptor objektumok prioritási sorát kezeli. Egyensúlyozza a számítási terhelést a magok között, miközben minimalizálja az adatmozgást az "adat-mag affinitás" tiszteletben tartásával.

Feladat végrehajtási logika:

1. Prioritási sor: A feladatok ArrayList-ben tárolódnak és priority és inference_priority szerint rendezve.
2. Affinitás leképezés: A DataFlowAnalyzer nyomon követi, hogy melyik magok melyik block_id-hez férnek hozzá.
3. Migráció: Ha a ChaosCoreKernel terhelési egyensúlyhiányt észlel (meghaladva a LOAD_HIGH_THRESHOLD-ot), rebalanceLoad()-ot indít, amely frissíti a blokkok affinity_core-ját és feladatokat mozgat az alulhasznált magokra.

Terheléselosztási konstansok:

- OPTIMIZATION_THRESHOLD: (0.6) Minimális nyereség a blokk migráció indításához.
- BALANCE_INTERVAL_CYCLES: (100) A terheléselosztó végrehajtásának gyakorisága.

executeGraphOnKernel interfész

Az executeGraphOnKernel függvény az elsődleges belépési pont komplex NSIR (Önhasonló Relációs Gráf) műveletek futtatásához a kernelen.

Végrehajtási folyamat:

1. Gráf elemzés: A kernel fogad egy SelfSimilarRelationalGraph-ot.
2. Feladat generálás: A csomópontok és élek TaskDescriptor egységekké konvertálódnak.
3. Függőség feloldás: Az adatfüggőségek CAS block_id-kre kerülnek leképezve a data_dependencies.append() segítségével.
4. Párhuzamos végrehajtás: A DynamicTaskScheduler feladatokat küld a RelationalGraphProcessingUnit-hoz (R-GPU) vagy helyi CPU szálakhoz a ChaosCoreConfig alapján.

---

4.4 CREV FOLYAMAT ÉS ZRUNTIME

A CREV (Oksági Érvelés és Ellenőrzés) folyamat és a ZRuntime végrehajtási motor alkotják a JAIDE rendszer mag kognitív feldolgozási rétegét. Míg a Neurális Feldolgozó Réteg (RSF) nagy dimenziós vektor transzformációkat kezel, a CREV/ZRuntime verem diszkrét relációs kivonást, oksági validációt és kvantum-relációs változó végrehajtást kezel.

1. CREV Folyamat

A CREV folyamat felelős a strukturálatlan természetes nyelv strukturált relációs hármasokká való átalakításáért és oksági konzisztenciájuk ellenőrzéséért a SelfSimilarRelationalGraph-on belül.

1.1 Kivonási szakaszok

A folyamat az ExtractionStage enum által definiált diszkrét szakaszok sorozatán keresztül működik:

| Szakasz | Leírás |
| :--- | :--- |
| tokenization | Kezdeti morfológiai és szó szintű szegmentálás. |
| triplet_extraction | Minta alapú Alany-Reláció-Tárgy (SRO) struktúrák azonosítása. |
| validation | Oksági lánc ellenőrzés és megbízhatósági pontozás. |
| integration | Validált hármasok összevonása az NSIR gráfba. |
| indexing | Relációs indexek frissítése a visszakereséshez. |

1.2 Hármas azonosság és hashelés

Az adatok integritásának és deduplikációjának biztosítása érdekében a CREV két hashelési stratégiát alkalmaz:

1. Azonosság hashelés: Sha256-ot alkalmaz a subject, relation és object mezőkön egy relációs tény egyedi azonosítójának generálásához.
2. Mező hashelés: Tartalmazza a confidence-t és extraction_time-ot a kivonás specifikus példányainak nyomon követéséhez.

2. ZRuntime Végrehajtási Motor

A ZRuntime a relációs logika végrehajtási környezete. Kezeli a ZVariable entitások életciklusát, amelyek szimbolikus változókat kvantum-relációs állapotokra képezik le.

2.1 ZVariable életciklus

Egy ZVariable egy SelfSimilarRelationalGraph-ot és egy RelationalQuantumLogic példányt foglal magában.

- Hozzárendelés (assign): Szimbolikus értéket köt a változóhoz, rögzítve azt a HistoryEntry naplóban.
- Reláció (relateTo): Élt hoz létre az aktuális változó és egy célváltozó között a gráfon belül.
- Mérés (measure): Összeomlasztja a változó kvantum állapotát egy diszkrét értékre, a RelationalQuantumLogic motort alkalmazva.

2.2 Relációs műveletek és kvantum kapuk

A relációs kifejezések elemzésre kerülnek és közvetlenül kvantum kapu műveletekre képezik le. A ZRuntime számos magas szintű operátort támogat:

| Relációs op | Kvantum leképezés | Implementáció |
| :--- | :--- | :--- |
| AND | Többvezérelt fázis | z_runtime.zig:78 |
| OR | Szuperpozíció / Hadamard | z_runtime.zig:78 |
| XOR | CNOT / Pauli-X | z_runtime.zig:78 |
| ENTANGLE | Bell állapot létrehozás | z_runtime.zig:290-299 |

2.3 Végrehajtási előzmények és auditálás

A ZRuntime-on belül végrehajtott minden művelet rögzítésre kerül egy ExecutionHistoryEntry-ben. Ez lehetővé teszi az érvelési folyamat teljes auditálhatóságát, beleértve:

- primary_target: A megcélzott változó.
- secondary_targets: Kapcsolódó változók (pl. összefonódásban).
- timestamp: Nanoszekundum pontosságú időzítés.
- result_value: Mérések vagy transzformációk eredménye.

3. Információ propagáció

Az információ propagáció a változó határain keresztül a propagateInformation segítségével kezelt. Ez a folyamat biztosítja, hogy amikor egy ZVariable állapot megváltozik (pl. mérés vagy külső hozzárendelés révén), a hatások az NSIR gráfon keresztül az összefonódott vagy kapcsolódó csomópontokra terjednek.

Propagációs folyamat:

1. Kiváltó: Állapotváltozás következik be a ZVariable A-ban.
2. Keresés: A ZRuntime azonosítja az összes élt a SelfSimilarRelationalGraph-ban, ahol A forrás.
3. Fázis eltolás: Az EdgeQuality (pl. entangled, coherent) meghatározza a jel propagáció nagyságát.
4. Frissítés: A célváltozó B fázis vagy amplitúdó beállítást kap a QuantumState-jében.
5. Audit: A propagációs esemény ExecutionAction.propagate_information-ként kerül naplózásra.

---

4.5 JELTERJEDÉS ÉS FNDS

A Jelterjedési Motor és a Fraktál Csomópont Adatrendszer (FNDS) biztosítják az NSIR gráf diszkrét idejű szimulációs és hierarchikus tárolási rétegeit. Míg a SelfSimilarRelationalGraph definiálja a topológiát, a Jelterjedési Motor szimulálja, hogyan áramlik az információ (hullámszerű állapotokként ábrázolva) az éleken keresztül, és az FNDS kezeli a csomópont-rezidens adatok tárolását és fraktál indexelését.

Jelterjedési Motor

A SignalPropagationEngine diszkrét idejű jelfolyamot szimulál a SelfSimilarRelationalGraph-on keresztül. A jelek SignalState objektumokként vannak modellezve, amelyek amplitúdót, fázist és frekvenciát tartalmaznak, lehetővé téve a rendszer számára az interferencia és rezonancia modellezését a relációs struktúrán belül.

Jel állapot és transzformáció

A jelek nem egyszerű skalárok; komplex értékű oszcillátorok.

- SignalState: Nyomon követi az amplitude-ot, phase-t és frequency-t.
- Temporális előrehaladás: Az advance függvény frissíti a jel fázisát a frekvenciája és az eltelt delta_time alapján.
- Komplex leképezés: A jelek komplex szám reprezentációvá konvertálhatók (A * e^(i*phi)) a getComplexRepresentation segítségével.

Propagációs logika

Amikor egy jel átmegy egy élen, az él tulajdonságai transzformálják:

1. Csillapítás: A jel amplitúdója az él súlyával csillapodik.
2. Fázis eltolás: Az él quantum_correlation tulajdonsága fázis eltolóként működik.
3. Qubit normalizálás: Ahogy a jelek aktiválják a csomópontokat, a csomópont belső Qubit állapota frissül és normalizálódik a normalizeNodeQubit segítségével.
4. Aktiváció rögzítés: Minden jel érkezés naplózásra kerül egy ActivationTrace-ben, amely signal_history-t tárol a temporális elemzéshez.

Fraktál Csomópont Adatrendszer (FNDS)

Az FNDS kezeli a csomópontokhoz kapcsolódó adatok hierarchikus, önhasonló tárolását. FractalTree-t alkalmaz az információ szervezéséhez oly módon, hogy lehetővé teszi a dobozszámolási dimenzió becslést és mintaillesztést különböző skálákon.

FractalNodeData

A FractalNodeData struktúra a csomópont tartalom elsődleges tárolója.

- Fraktál aláírás: SHA-256 hash, amelyet a csomópont azonosítójából, adataiból, súlyából és skálájából számítanak.
- Metaadat: StringHashMap tetszőleges kulcs-érték párokhoz a csomóponthoz kapcsolódóan.
- Önhasonlóság: A csomópontok nyomon követik scale-jüket és children_count-jukat a fraktál elemzés támogatásához.

FractalTree és FNDSManager

Az FNDSManager több FractalTree példányt koordinál.

- Dobozszámolási dimenzió: A rendszer képes becsülni az adateloszlás komplexitását a boxCountingDimension segítségével.
- Mintaillesztés: A SelfSimilarIndex mintaillesztés lehetővé teszi a rendszer számára, hogy megtalálja azokat a részfákat, amelyek megfelelnek egy specifikus strukturális vagy adat aláírásnak.
- Hatékonyság: A manager CoalescedHashMap-et alkalmaz a sűrű tároláshoz és LRUCache-t a drága fraktál újraszámítások minimalizálásához.

Főbb implementációs részletek

Jel motor statisztikák

A PropagationStatistics struktúra nyomon követi a jel kitörések hatékonyságát és elérését:

- total_activations: Minden alkalommal, amikor egy csomópont küszöbértéke teljesült.
- unique_nodes_activated: A jel diszperzió mértéke.
- average_propagation_speed: A time_step és gráf távolság alapján számítva.

FNDS hibakezelés

Az FNDS specifikus FNDSError készletet alkalmaz a fraktál-specifikus meghibásodási módok kezeléséhez:

- PatternLengthOutOfRange: SelfSimilarIndex keresések során fordul elő.
- InvalidScale / InvalidWeight: Akkor aktiválódik, ha nem véges (NaN/Inf) értékek kerülnek átadásra a FractalNodeData.init-nek.
- CycleDetected: Kritikus hiba, amikor egy fraktál fa struktúra megsérti az Irányított Aciklikus Gráf (DAG) követelményt.

Teljesítmény segédprogramok:

- Telítési aritmetika: A satAddUsize és satSubUsize függvények a statisztika követésben kerülnek alkalmazásra a hosszú futású szimulációkban való túlcsordulás megelőzéséhez.
- Kanonikus floatok: A canonicalF64Bytes biztosítja, hogy a NaN és nulla értékek determinisztikus bájt reprezentációval rendelkezzenek a fraktál aláírásba való hashelés előtt.

---

4.6 MEGLEPETÉS MEMÓRIA ÉS TEMPORÁLIS GRÁF

A Meglepetés Memória és Temporális Gráf alrendszerek biztosítják a JAIDE motor számára az újszerű információ azonosítását és a relációs adatok időbeli fejlődésének nyomon követését. A SurpriseMemoryManager Jaccard és Hamming metrikák alapján szűri az adatokat az újdonságuk szerint, míg a TemporalGraph nanoszekundum pontosságú pillanatképeket tart fenn az NSIR gráf állapotáról.

Meglepetés Memória Manager

A SurpriseMemoryManager kapuőrként működik a Tartalom-Címezhető Tárolás (CAS) számára. Értékeli a bejövő adatblokkokat annak meghatározásához, hogy elegendő "meglepetést" (újdonságot) tartalmaznak-e a relációs magba való hosszú távú rögzítés indoklásához.

Újdonság metrikák és pontozás

A meglepetés három elsődleges metrika kombinációján keresztül kerül kiszámításra, amelyek a SurpriseMetrics struktúrában vannak összefoglalva:

1. Jaccard disszimilaritás: Méri a bigram készletek átfedését az új adatok és a meglévő minták között.
2. Tartalom hash távolság: Kiszámítja a Hamming távolságot az új blokk SHA-256 hash-e és a meglévő blokk azonosítók között.
3. Temporális újdonság: Értékeli, hogy mikor dolgoztak fel hasonló adatokat utoljára egy 86 400 másodperces csúszó ablak segítségével (TEMPORAL_NOVELTY_WINDOW_NS).

A kombinált meglepetési pontszám ezen tényezők normalizált átlaga.

Megőrzés és kiürítés

A meglepetés memóriába rögzített adatok SurpriseRecord-ot kapnak. A rendszer súlyozott prioritási algoritmust alkalmaz ezek életciklusának kezeléséhez:

| Konstans | Érték | Leírás |
| :--- | :--- | :--- |
| RETENTION_BASE_WEIGHT | 0.5 | Alapsúly minden rekordhoz. |
| RETENTION_AGE_WEIGHT | 0.3 | Csökkentési tényező a rekord kora alapján. |
| RETENTION_FREQUENCY_WEIGHT | 0.2 | Erősítési tényező a gyakran hozzáférhető blokkokhoz. |

A recomputeRetention függvény frissíti a retention_priority-t minden alkalommal, amikor egy blokkhoz hozzáférnek vagy periodikus karbantartás során.

Temporális Gráf

A TemporalGraph verzionált, idősor-alapú nézetet biztosít a SelfSimilarRelationalGraph-ról (NSIR). Nanoszekundum felbontással követi nyomon a csomópontok és élek változásait.

Verzionálási primitívek

A rendszer két elsődleges verzionálási struktúrát definiál:

- NodeVersion: Tárolja a csomópont QuantumState pillanatképét, verziószámát és egy StringHashMap-et a tulajdonságokról egy specifikus Timestamp-nél.
- EdgeVersion: Nyomon követi a két csomópont közötti kapcsolat weight-jét és EdgeQuality-jét (pl. Coherent, Entangled, Fractal) egy adott időbélyegnél.

Kvantum állapot pillanatképek

A NodeVersion kifejezetten rögzíti a QuantumState-et a quantum_logic modulból. Ez lehetővé teszi a ReasoningOrchestrator számára az "időbeli visszagörgetések" végrehajtását, ahol lekérdezheti egy csomópont valószínűségét és nagyságát a tanítási előzmények bármely pontján.

Főbb függvények és konstansok

| Entitás | Elhelyezkedés | Cél |
| :--- | :--- | :--- |
| Timestamp | src/core_relational/temporal_graph.zig:14 | i64 nanoszekundum reprezentáció. |
| NodeVersion.init | src/core_relational/temporal_graph.zig:42-55 | Új csomópont pillanatképet hoz létre QuantumState-tel. |
| EdgeVersion.init | src/core_relational/temporal_graph.zig:155-170 | Új él pillanatképet hoz létre EdgeQuality-vel. |
| SurpriseRecord.recomputeRetention | src/core_relational/surprise_memory.zig:88-97 | Újraszámítja a prioritást RETENTION_AGE_WEIGHT segítségével. |
| SurpriseMetrics.init | src/core_relational/surprise_memory.zig:61-72 | Kombinálja a Jaccard, Hash és Temporális pontszámokat. |

Memóriakezelés és stabilitás

A SurpriseMemoryManager Mutex-et alkalmaz a meglepetés rekordokhoz és statisztikákhoz való szálbiztos hozzáférés biztosítására. Az időbeli konzisztencia fenntartásához a stableMonotonicNow függvény megakadályozza, hogy az óra eltolódása vagy a rendszeridő módosítása visszafelé mozgó időbélyegeket okozzon a memória rekordokban.

---

5 KVANTUM SZÁMÍTÁSTECHNIKAI INTEGRÁCIÓ

A JAIDE kvantum számítástechnikai primitíveket integrál közvetlenül a kognitív architektúrájába, áthidalva a klasszikus neurális feldolgozás és a kvantum-relációs érvelés közötti szakadékot. Ez a réteg biztosítja a szükséges absztrakciókat a kvantum áramkörök fizikai hardveren (IBM Quantum) vagy szimulátorokban való végrehajtásához, miközben egy Nulla-Tudás (ZK) ellenőrzési rendszeren keresztül biztosítja az eredmények integritását.

A kvantum réteget elsősorban a Mag Relációs Réteg használja a tokenek és fogalmak közötti komplex összefonódások modellezéséhez a SelfSimilarRelationalGraph-on (NSIR) belül.

Rendszer áttekintés

Az integráció három elsődleges tartományból áll:

1. Kvantum Logika Motor: Kvantum állapotokat és kapukat szimulál és kezel a JAIDE futtatókörnyezetben.
2. Hardver Interfész: Külső kvantum backendekkel (IBM Quantum) és helyi szimulátorokban való kommunikációt kezel.
3. ZK Ellenőrzés: Kriptográfiai bizonyítékokat biztosít arról, hogy a kvantum-klasszikus hibrid következtetések helyesen kerültek végrehajtásra az érzékeny modell paraméterek felfedése nélkül.

Kvantum Logika és Hardver Interfész

A JAIDE átfogó kvantum logikai kapu készletet valósít meg, amely túlmutat a szabványos qubiteken és relációs műveleteket is tartalmaz. A LogicGate enum mind a szabványos kapukat (Hadamard, CNOT, Toffoli), mind a JAIDE-specifikus relációs primitíveket definiálja, mint a RELATIONAL_AND és a FRACTAL_TRANSFORM.

A QuantumState struktúra kezeli a komplex amplitúdókat és fázis információkat ezekhez a műveletekhez, segédprogramokat biztosítva a normalizáláshoz és a valószínűség számításhoz. A fizikai végrehajtáshoz az IBMQuantumClient kezeli a QuantumCircuit életciklusát, az OpenQASM 3.0 szerializálástól az IBM hardver családokon (HERON, EAGLE vagy FALCON) való benyújtásig.

Főbb komponensek:

- RelationalQuantumLogic: Kapu alkalmazásokat orchestrál az NSIR gráfon.
- IBMQuantumClient: Kezeli a backend kalibrációs adatokat (T1/T2 idők) és a feladat sorba állítást.
- Hibrid Optimalizáló: Paraméter-eltolás gradienseket alkalmaz a kvantum-klasszikus paraméterek hangolásához.

Nulla-Tudás Ellenőrzési Rendszer

A hibrid kvantum-neurális következtetések biztonságának és helyességének biztosítása érdekében a JAIDE Nulla-Tudás (ZK) ellenőrzési réteget alkalmaz. Ez a rendszer, amelynek középpontjában a ZKInferenceProver és a VerifiedInferenceEngine áll, Groth16 bizonyítékokat generál a bn128 görbe segítségével.

A rendszer circom eszközláncot alkalmaz egy inference_trace.circom áramkör fordításához, amely validálja a következtetési folyamat Poseidon-láncát. Ez lehetővé teszi a JAIDE számára, hogy bizonyítsa, hogy egy specifikus kimenetet egy specifikus modell és bemenet generált, anélkül, hogy felfedné az alapul szolgáló Tensor súlyokat vagy az NSIR gráf topológiát.

Főbb jellemzők:

- Differenciális Adatvédelem: Laplace/Gauss zaj injektálása az adathalmaz adatvédelmének védelméhez.
- Rögzített Pontos Skálázás: Az InferenceWitness kezeli a JAIDE Fixed32_32 aritmetikája és a ZK-barát prímtestek közötti konverziót.
- Biztonságos Aggregáció: Merkle-fa alapú bizonyíték aggregációt tesz lehetővé a nagy áteresztőképességű köteg ellenőrzéshez.

Integrációs logika

A ZRuntime végrehajtási motorként szolgál, amely összeköti ezeket a komponenseket. ExecutionAction parancsokat dolgoz fel, mint a quantum_circuit vagy az entangle_variables, és azokat a helyi RelationalQuantumLogic szimulátorhoz vagy az IBMQuantumClient-hez irányítja. Az eredmények ezután a VerifiedInferenceEngine-en keresztül kerülnek feldolgozásra a végső, kriptográfiailag biztosított válasz generálásához.

| Jellemző | Kód entitás | Fájl |
| :--- | :--- | :--- |
| Relációs kapuk | LogicGate.RELATIONAL_XOR | src/core_relational/quantum_logic.zig |
| Állapotkezelés | QuantumState | src/core_relational/quantum_logic.zig |
| Végrehajtási motor | ZRuntime | README.md |
| Hardver híd | IBMQuantumClient | README.md |
| ZK Bizonyítás | ZKInferenceProver | README.md |

---

5.1 KVANTUM LOGIKA ÉS IBM HARDVER INTERFÉSZ

A Kvantum Logika és IBM Hardver Interfész hidat biztosít a JAIDE rendszer relációs kognitív struktúrái és a fizikai kvantum számítás között. Magában foglalja a RelationalQuantumLogic motort a helyi szimulációhoz és az IBMQuantumClient-et a valós hardveren való végrehajtáshoz.

RelationalQuantumLogic Motor

A RelationalQuantumLogic motor felelős a relációs műveletek kvantum kapukra való leképezéséért és a kvantum állapotok életciklusának kezeléséért. Szabványos kvantum primitívek és speciális relációs kapuk készletét biztosítja.

Kapu műveletek

A rendszer átfogó logikai kapu készletet definiál a LogicGate enumban. Ezek tartalmazzák:

- Szabványos kapuk: HADAMARD, PAULI_X/Y/Z, PHASE, CNOT, TOFFOLI.
- Relációs kapuk: RELATIONAL_AND, RELATIONAL_OR, RELATIONAL_NOT, RELATIONAL_XOR.
- Speciális kapuk: FRACTAL_TRANSFORM az önhasonló állapot keveréshez.

A motor a kapukat qubit követelményeik szerint osztályozza és támogatja az egykubites műveleteket a többkubites összefonódó műveletekkel szemben.

Kvantum állapot reprezentáció

A kvantum állapotokat a QuantumState struktúra képviseli, amely nyomon követi:

- Amplitúdók: Komplex számok 2 elemű tömbje, amely az állapot vektort képviseli.
- Fázis: A qubit globális fázisa.
- Összefonódási fok: Skaláris érték, amely a más csomópontokkal való korrelációs erősséget képviseli.

IBM Quantum Hardver Interfész

Az IBMQuantumClient kezeli az IBM Quantum Platformmal való kommunikációt REST API-n keresztül.

Backend családok és kalibráció

A rendszer több IBM hardver családot támogat, előre definiált specifikációkkal és hibaprofillal az IBMBackendSpecs-ben:

- HERON: 133 qubit, T1 kb. 350 mikroszekundum.
- EAGLE: 127 qubit, T1 kb. 200 mikroszekundum.
- FALCON: 27 qubit, T1 kb. 100 mikroszekundum.

Az IBMBackendCalibrationData struktúra valós idejű telemetriát tárol, beleértve a T1/T2 időket, leolvasási hibákat és kapu hibákat a kiválasztott backend minden qubitjéhez.

Feladat benyújtás és OpenQASM

A kliens kezeli a kvantum feladat teljes életciklusát:

1. Szerializáció: Az áramkörök OpenQASM 3.0 karakterláncokká konvertálódnak.
2. Benyújtás: A submitJobWithBackend POST kérést hajt végre az IBM Cloud API-hoz.
3. Lekérdezés: Az eredmények a getJobResult segítségével kerülnek visszanyerésre a visszaadott feladat azonosító alapján.

Szimuláció és hibrid optimalizálás

Állapotvektor szimulátor zajmodellezéssel

Ha a use_real_backend hamis, a QuantumTaskAdapter a local_simulator-t alkalmazza. Ez a szimulátor megvalósítja:

- Zajmodellezés: Az IBMDocumentedBackendSpecs kalibrációs adatait alkalmazza a dekoherencia (T1/T2) és kapu hűtlenségek szimulálásához.
- Korlátok: A szimuláció 32 qubitre van korlátozva (SIMULATOR_QUBITS).

Kvantum-Klasszikus Hibrid Optimalizáló

A rendszer hibrid algoritmusokat (VQE, QAOA) támogat egy QuantumClassicalHybridOptimizer-en keresztül.

- Paraméter-eltolás gradiensek: Gradienseket számít a kvantum áramkör paraméterek eltolásával (pl. rotációs szögek) az objektív függvény optimalizálásához klasszikus hardveren.
- Konfiguráció: Az alapértelmezések 0.1-es tanulási rátát és 10^-6 toleranciát tartalmaznak.

Főbb konstansok összefoglalója

| Paraméter | Érték | Leírás |
| :--- | :--- | :--- |
| HERON_QUBITS | 133 | Max qubitek Heron osztályú hardverhez |
| SIMULATOR_QUBITS | 32 | Max qubitek helyi állapotvektor szimulációhoz |
| HARDWARE_MAX_SHOTS | 100 000 | Maximális mintavételi lövések áramkörenként |
| POLL_INTERVAL_MS | 100 | Lekérdezési frekvencia a feladat eredményekhez |

---

5.2 NULLA-TUDÁS ELLENŐRZÉSI RENDSZER

A JAIDE Nulla-Tudás (ZK) Ellenőrzési Rendszere mechanizmust biztosít az ellenőrizhető következtetéshez, biztosítva, hogy a neurális hálózati számítások és a relációs gráf átmenetek helyesen kerültek végrehajtásra anélkül, hogy felfednék az alapul szolgáló modell súlyokat vagy az érzékeny bemeneti adatokat. A Groth16 bizonyítási rendszert alkalmazza a bn128 elliptikus görbe felett, egyedi Circom-alapú eszközláncot alkalmazva az áramkör generáláshoz és snarkjs-t a bizonyíték orchestráláshoz.

Rendszer architektúra

A ZK rendszer egy magas szintű Zig interfészre (ZKInferenceProver) és egy alacsony szintű R1CS áramkör definícióra (inference_trace.circom) van osztva. Az architektúra "kötelezd el-majd-bizonyítsd" mintát követ, ahol a bemenetek és kimenetek Blake3 vagy Poseidon hash-ekkel kerülnek elkötelezésre, és a bizonyíték validálja az ezen elkötelezések közötti átmenetet.

CircomProver és eszközlánc integráció

A CircomProver osztály hídként működik a Zig futtatókörnyezet és a snarkjs/circom eszközlánc között. Kezeli az áramkör fordítást, a megbízható beállítást (Groth16) és a tanú generálást.

Főbb függvények:

- compileCircuit(): circom folyamatot indít az R1CS és WASM artifaktumok generálásához.
- generateWitness(): A lefordított WASM-t és node-ot alkalmazza a tanú kiszámításához a bemeneti jelekből.
- prove(): Végrehajtja az snarkjs groth16 prove-t egy ZKProofBundle létrehozásához, amely tartalmazza a Groth16Proof-ot és a PublicSignals-t.

Adatstruktúrák

| Struktúra | Cél |
| :--- | :--- |
| ZKCircuitConfig | Definiálja a .wasm, .zkey útvonalakat és a pontossági paramétereket (alapértelmezett 64 bites). |
| Groth16Proof | Magában foglalja a G1 és G2 pontokat (pi_a, pi_b, pi_c) a bn128-hoz. |
| PublicSignals | i256 értékek gyűjteménye, amelyek az áramkör nyilvános bemeneteit/kimeneteit képviselik. |

Következtetési nyom áramkör (inference_trace.circom)

A ZK rendszer mag logikája az inference_trace.circom-ban található. Rögzített pontos aritmetikát és specifikus neurális rétegeket valósít meg ZK-barát módon.

Poseidon láncolás

Mivel a szabványos hash-ek, mint az SHA-256, drágák az R1CS-ben, a JAIDE PoseidonChain(n)-t alkalmaz az állapot elkötelezésekhez. A bemeneteket 6-os darabokban dolgozza fel, Poseidon hash függvényeken láncolva azokat egyetlen mezőelem kimenet előállításához.

RSF réteg ellenőrzés

Az RSFLayerComputation(dim) sablon tükrözi az RSFLayer-t a neurális veremben. Validálja:

1. Osztás: Az x bemeneti vektor x1-re és x2-re osztódik.
2. Affin csatolás: y2 = x2 ⊙ exp(S(x1)) + T(x1).
3. Rögzített pontos skálázás: Mivel a Circom véges testekben dolgozik, a floatok FIXED_POINT_SCALE (10^6) segítségével skálázódnak.
4. Taylor közelítés: Az exp függvény köbös Taylor sorral közelítendő: 1 + x + 0.5x^2 + 0.166667x^3.

Tartomány és tagság bizonyítékok

- RangeProof(bits): Biztosítja, hogy egy érték [min, max] tartományban legyen Num2Bits dekompozíció és Pedersen elkötelezések segítségével minden bithez.
- VerifyMerkleProof(depth): Szabványos Merkle fa útvonal validálást valósít meg Poseidon(2) hashelők és Mux1 segítségével az útvonal index váltáshoz.

Adatvédelem és ellenőrzési logika

A rendszer Differenciális Adatvédelmet és Biztonságos Aggregációt tartalmaz az egyéni adatpontok védelmére az ellenőrzési folyamat során.

Differenciális adatvédelem

A ZKInferenceProver zajt alkalmaz a következtetési nyomra az (ε, δ)-differenciális adatvédelem teljesítéséhez.

- Laplace zaj: SecureRng segítségével generálva és a nyilvános jelekbe injektálva a pontos értékek elhomályosításához.
- Gauss zaj: Magasabb dimenziós aggregációkhoz alkalmazva.

Biztonságos aggregáció

Az elosztott következtetéshez a rendszer támogatja a bizonyítékok aggregálását több résztvevőtől.

- SecureAggregation: Biztosítja, hogy az aggregált eredmény helyes legyen az egyéni hozzájárulások felfedése nélkül.
- Blake3 elkötelezés: Nagy sebességű bemenet/kimenet integritás ellenőrzéshez alkalmazva a drágább ZK bizonyíték generálása előtt.

Hibakezelés

A ZKProofError enum definiálja az ellenőrzési folyamat meghibásodási módjait, beleértve a CircomCompilationFailed, WitnessGenerationFailed és SnarkjsNotFound hibákat. Ezek a hibák a VerifiedInferenceEngine-en keresztül propagálódnak annak biztosítására, hogy az ellenőrizetlen eredmények soha ne kerüljenek érvényesként kezelésre magas integritású módokban.

---

6 KÖVETKEZTETÉSI SZERVER ÉS VISSZAKERESÉS

A Következtetési Szerver és Visszakeresési réteg az elsődleges interfészként szolgál a külső fogyasztók számára a JAIDE rendszerrel való interakcióhoz. Orchestrálja az átmenetet a nyers szöveges bemenetektől a nagy dimenziós neurális reprezentációkig, a relációs gráf érvelésig és végül a token generálásig. Ez a réteg kezeli a HTTP kapcsolatok életciklusát, érvényesíti a biztonsági és sebességkorlátozásokat, és speciális indexelési struktúrákat (SSI) és rangsorolási algoritmusokat alkalmaz a kontextus és koherencia fenntartásához a következtetés során.

Kiszolgálási architektúra áttekintés

Az InferenceServer egy többszálú HTTP motor, amelyet nagy áteresztőképességű token generálásra terveztek. ThreadPool-t alkalmaz az egyidejű kapcsolatok kezeléséhez, inference_mutex-szel védve a szálbiztos hozzáférés biztosítása érdekében az alapul szolgáló modell súlyokhoz és állapothoz.

A szerver szabványos RESTful végpontokat valósít meg, beleértve a /v1/health-et a monitorozáshoz és a /v1/inference-t az egyszeri kérés feldolgozáshoz. Nagy sűrűségű munkaterhelésekhez a /v1/batch_inference végpont lehetővé teszi több prompt párhuzamos feldolgozását.

A következtetési folyamat

Amikor egy kérés érkezik, a szerver komplex folyamatot hajt végre, amely áthidalja a diszkrét tokenek és a Mag Relációs Réteg közötti szakadékot.

| Fázis | Komponens | Művelet |
| :--- | :--- | :--- |
| Belépés | RateLimiter | IP/API kulcs validálása max_requests_per_minute ellen. |
| Tokenizálás | MGT | Nyers szöveg szódarab egységekké konvertálása. |
| Neurális folyam | RSFLayer | Beágyazások átadása Visszafordítható Szórt Folyam rétegeken. |
| Érvelés | ReasoningOrchestrator | Hierarchikus érvelés indítása (helyi/globális/meta). |
| Visszakeresés | SSI és Ranker | Szegmentált Szekvencia Index lekérdezése releváns kontextushoz. |
| Generálás | FractalLPU | Token generálási ciklus végrehajtása hardver gyorsítókon. |

Visszakeresés és kontextus rangsorolás

A JAIDE Szegmentált Szekvencia Indexet (SSI) alkalmaz a neurális állapotok és relációs hármasok kereshető előzményének fenntartásához. Az SSI hierarchikus hash faként van strukturálva, lehetővé téve a retrieveTopK hasonlósági kereséseket, amelyek tájékoztatják a Ranker-t. A Ranker n-gram csökkentési súlyozást és Jaccard hasonlóságot alkalmaz a potenciális következő tokenek pontozásához, biztosítva, hogy a generált kimenet a megadott kontextusban és a modell belső memóriájában maradjon.

Adatfolyam: Szövegtől a relációs állapotig

Az InferenceServer a CPU-kötött API logika és a GPU/LPU-kötött neurális számítások koordinátoraként működik. A VerifiedInferenceEngine-t alkalmazza annak biztosítására, hogy a felhasználónak visszaadott eredmények kriptográfiailag konzisztensek legyenek a modell állapotával.

---

6.1 HTTP KÖVETKEZTETÉSI SZERVER

Az InferenceServer a JAIDE kiszolgálási rétegének elsődleges belépési pontja, nagy teljesítményű HTTP interfészt biztosítva mind az egyszeri, mind a köteg következtetéshez. Orchestrálja a komplex átmenetet a természetes nyelvi bemenetektől az RSF neurális vermen és a mag relációs érvelési motoron keresztül.

Szerver architektúra

Az InferenceServer dedikált ThreadPool-ra épülő többszálú, aszinkron architektúrán alapul a kapcsolat életciklusok kezeléséhez a fő eseményhurok blokkolása nélkül.

Főbb komponensek:

- RateLimiter: Csúszóablak algoritmust valósít meg az IP-cím szerinti kérések nyomon követéséhez.
- Következtetési Mutex: Egy globális Thread.Mutex biztosítja a szálbiztos hozzáférést az alapul szolgáló modell súlyokhoz és a SelfSimilarRelationalGraph-hoz az előre irányuló menet során.
- Kapcsolat életciklus: Minden bejövő kapcsolatot a fő szál fogad és a poolhoz irányít, ahol API kulcsokra kerül validálásra (ha a require_api_key engedélyezve van).

API végpontok

A szerver három elsődleges REST végpontot tesz elérhetővé:

| Végpont | Módszer | Leírás |
| :--- | :--- | :--- |
| /v1/health | GET | Visszaadja a szerver állapotát, üzemidejét és modell betöltési állapotát. |
| /v1/inference | POST | Szabványos egyszeri kérés következtetés. InferenceRequest JSON-t vár. |
| /v1/batch_inference | POST | Több prompt párhuzamos feldolgozása ServerConfig.batch_size-ig. |

A következtetési folyamat

A szerver magja a runInferenceInternal függvény, amely végrehajtja a teljes transzformációt a tokenektől a relációs érvelésig és vissza a generált szövegig.

Adatfolyam szakaszok:

1. Tokenizálás: Az MGT (Multi-Gram Tokenizáló) a bemeneti szöveget token azonosítók sorozatává konvertálja.
2. Beágyazás: A tokenek nagy dimenziós térbe kerülnek vetítve a LearnedEmbedding segítségével.
3. RSF előre irányuló menet: Az RSFLayer verem visszafordítható affin csatolást és OFTB keverést hajt végre.
4. NSIR kódolás: A neurális állapot kódolódik a SelfSimilarRelationalGraph-ba (NSIR).
5. Érvelés orchestrálás: A ReasoningOrchestrator futtatja a háromfázisú érvelési ciklust (helyi, globális, meta).
6. Hardver gyorsítás: A munkaterhelések a FractalLPU-hoz és a RelationalGraphProcessingUnit-hoz (R-GPU) kerülnek irányítva gráf feldolgozáshoz.
7. Meglepetés memória: Az újszerű minták a SurpriseMemoryManager-be kerülnek rögzítésre hosszú távú megőrzésre.
8. Token generálás: A végső állapot visszadekódolódik az RSF inverz útvonalon a következő token mintavételezéséhez.

Ellenőrzött következtetés integráció

A szerver "Ellenőrzött" módot támogat a VerifiedInferenceEngine-en keresztül. Ha engedélyezve van, a szerver Nulla-Tudás (ZK) bizonyítékot generál a következtetés végrehajtásáról.

- Elkötelezés: A bemeneti tokenek és modell súlyok Blake3 segítségével kerülnek hash-elve egy elkötelezés létrehozásához.
- Nyom rögzítés: A ReasoningOrchestrator minden művelete rögzítésre kerül egy InferenceWitness-be.
- Bizonyíték generálás: A kérés befejezésekor a VerifiedInferenceEngine a CircomProver-t alkalmazza egy Groth16 bizonyíték generálásához, amely igazolja, hogy a kimenet helyesen lett levezetva az elkötelezett bemenetből és modellből.

Szerver konfiguráció és inicializálás

A szerver a ServerConfig struktúrán keresztül kerül konfigurálásra. Az inference_server_main.zig fő belépési pontján keresztül inicializálható és indítható.

Konfigurációs paraméterek:

- batch_size: Meghatározza az egyidejűleg feldolgozható szekvenciák maximális számát az RSF veremben.
- esso_initial_temp: Szabályozza az EntangledStochasticSymmetryOptimizer kezdeti hőmérsékletét az érvelési fázis során.
- require_api_key: Logikai jelző a hitelesítés érvényesítéséhez.

Fő végrehajtás

A main függvény kezeli a parancssori argumentum elemzést, a környezeti változó felülírásokat (pl. JAIDE_MODEL_PATH) és a kecses leállítást.

---

6.2 SSI INDEX ÉS RANKER

A Szegmentált Szekvencia Index (SSI) és a Ranker alrendszerek biztosítják a JAIDE következtetési folyamat alapvető visszakeresési és pontozási infrastruktúráját. Az SSI hierarchikus hash fát valósít meg a token szegmensek hatékony tárolásához és integritás-ellenőrzött visszakereséséhez, míg a Ranker többtényezős pontozást biztosít n-gram csökkentéssel, MinHash-alapú Jaccard hasonlósággal és diverzitási metrikákkal.

SSI: Szegmentált Szekvencia Index

Az SSI egy hierarchikus hash fa struktúra, amelyet token hash-ek alapján Segment adatok tárolására és visszakeresésére terveztek. Tartalom-címezhető indexként működik Merkle-stílusú integritás ellenőrzésekkel és automatikus egyensúlyozással.

Adatstruktúrák és hierarchia

Az index Node objektumok fájává van szervezve, ahol minden csomópont lehet ág (gyermekeket tartalmaz) vagy levél (szegmenseket és ütközési láncokat tartalmaz).

- Segment: Token sorozatot képvisel kapcsolódó metaadatokkal, beleértve egy globális position-t, score-t és anchor_hash-t.
- Node: Tartalmaz egy hash-t, amely a részfájának állapotát képviseli, children listát (ágakhoz) és segment-et vagy collision_chain-t (levelekhez).
- CollisionNode: Láncolt lista struktúra a levél csomópontokon belüli hash ütközések kezeléséhez.

SSI implementációs logika

Az SSI 6-os bucket_width-et alkalmaz, ami 64 gyermeket eredményez ág csomópontonként. A hash integritást a refreshHash tartja fenn, amely egy csomópont hash-ét a gyermekei (ágakhoz) vagy szegmensei (levelekhez) alapján számítja.

| Jellemző | Implementációs részlet |
| :--- | :--- |
| Hash algoritmus | Egyedi mixHash 0x9E3779B185EBCA87 konstanssal |
| Integritás | Merkle-stílusú rekurzív hashelés computeBranchHash-en keresztül |
| Ütközés kezelés | Láncolt lista collision_chain levél csomópontokban |
| Keresés | retrieveTopK hasonlósági keresés szegmens pontszámok alapján |

Ranker: Szekvencia pontozás és visszakeresés

A Ranker felelős a token szekvenciák relevanciájának és minőségének értékeléséért. N-gram súlyok, Lokalitás-Érzékeny Hashelés (LSH) és diverzitási pontozás kombinációját alkalmazza normalizált pontszám előállításához.

Pontozási komponensek

A Ranker több súlyozott tényezőn keresztül számítja a pontszámokat a RankerConfig-ban definiálva:

1. N-gram csökkentés: Súlyok kerülnek hozzárendelésre az n-gramokhoz 1/N csökkentési mintával, a hosszabb egyezéseket részesítve előnyben.
2. Diverzitási pontozás: A computeTokenDiversity kiszámítja az egyedi tokenek és az összes token arányát az ismétlődő szekvenciák büntetéséhez.
3. Horgony közelség: Méri, hogy a tokenek mennyire közel vannak az SSI-n belüli ismert horgony hash-ekhez.
4. Jaccard hasonlóság: MinHash aláírásokat alkalmaz a szekvencia és egy lekérdezés közötti hasonlóság becslésére.

Ranker logika folyam

A pontozás elsődleges belépési pontja a scoreSequence, amely összesíti a tényezőket nyers pontszámmá és 0.0 és 1.0 közé szorítja. A lekérdezés alapú visszakereséshez a scoreSequenceWithQuery kombinálja az alap szekvencia pontszámot az átfedési és Jaccard metrikákkal.

Tenzor integráció és perzisztencia

Az SSI index támogatja a struktúrájának exportálását és importálását a JAIDE Tensor formátumba a perzisztencia és GPU-gyorsított feldolgozás érdekében.

- Export: Az exportToTensor szerializálja a fát egy 2D tenzorrá [méret, 134] alakban, ahol 134 a tensor_width. Ez a szélesség befogadja a Segment metaadatokat és token hash-eket.
- Import: Az importFromTensor rekonstruálja a hierarchikus Node struktúrát egy szerializált tenzorból, validálva az anchor_hash-t és rekonstruálva a collision_chain-t minden levélhez.

Tenzor export séma

| Eltolás | Mező | Leírás |
| :--- | :--- | :--- |
| 0 | Pozíció | Globális szekvencia pozíció (u64 két f32-re osztva) |
| 2 | Pontszám | Lebegőpontos szegmens pontszám |
| 3 | Horgony hash | Hash a közelség nyomon követéséhez (u64 osztva) |
| 5-133 | Tokenek | Token azonosító tárolás (legfeljebb 128 token) |

Fejlett visszakeresési jellemzők

Top-K visszakeresés

A Ranker topKHeap visszakeresést valósít meg, amely prioritási sort tart fenn a legmagasabb pontszámú szegmensekből egy streaming rangsorolási művelet során. Ez lehetővé teszi a rendszer számára a nagy léptékű index bejárások kezelését kimerítő rendezés nélkül.

Párhuzamos pontozás

A Ranker párhuzamos végrehajtásra van tervezve. A pontozási menetek szálak között oszthatók el, a súly kalibrálás periodikusan történik az ngram_weights beállításához a ReasoningOrchestrator visszajelzése alapján.

Streaming rangsorolás

Valós idejű következtetéshez a Ranker streaming rangsorolási módot támogat. STREAMING_WINDOW_SIZE-t (512 token) alkalmaz a pontszámok inkrementális kiszámításához, ahogy a tokenek generálódnak az RSF neurális verem által.

---

7 HARDVER GYORSÍTÁS

A JAIDE heterogén hardver gyorsítási vermet alkalmaz az 5. gyök architektúra számítási igényeinek kezeléséhez. A rendszer optimalizált GPU kerneleken, egyedi logikai egységeken és RTL szintű hardver leírásokon keresztül hidalja át a magas szintű neurális műveleteket és a relációs gráf feldolgozást.

A verem három elsődleges szintre van osztva:

1. GPU gyorsítás (Futhark/CUDA): Nagy áteresztőképességű neurális műveletek az RSF (Visszafordítható Szórt Folyam) rétegekhez.
2. Relációs feldolgozás (FractalLPU/R-GPU): Speciális architektúrák a gráf alapú megismeréshez és Network-on-Chip (NoC) szimulációhoz.
3. RTL logika (Clash/Haskell): Alacsony szintű hardver modulok a memória arbitrációhoz és a nagy sebességű visszakereséshez.

---

7.1 FUTHARK GYORSÍTÓ ÉS CUDA INTERFÉSZ

A JAIDE hardver gyorsítási rétege nagy teljesítményű interfészt biztosít a neurális műveletekhez Futhark által generált GPU kernelek és nyers CUDA kötések segítségével. Ez a rendszer kezeli a GPU kontextusok életciklusát, típusbiztos burkolókat biztosít a többdimenziós eszköz tömbökhoz, és megvalósítja a Visszafordítható Szórt Folyam (RSF) modellek tanítási folyamatát.

RSFAccelerator és kontextus életciklus

A FutharkContext struktúra az elsődleges kezelő a GPU erőforrásokhoz, burkolva a Futhark által generált C API-t. Kezeli a futhark_context-et és a hozzá tartozó konfigurációt.

Kontextus inicializálás

Inicializáláskor a gyorsító konfigurálja a GPU eszközt és beállítja az alapértelmezett végrehajtási paramétereket:

- Csoport méret: 256 szál blokkonként.
- Csoportok száma: 128 blokk.
- Csempe méret: 32x32 mátrix műveletekhez.

A sync() függvény biztosítja, hogy az összes aszinkron GPU kernel befejeződjön, mielőtt a gazdagép folytatja, ami kritikus az adatok integritásának fenntartásához a tanítási lépések során.

FutharkArray típus burkolók

A JAIDE speciális burkolókat alkalmaz az eszközön tárolt tömbökhoz a típusbiztonság és a helyes dimenzionalitás biztosítása érdekében a Futhark bejegyzések hívásakor. Ezek a burkolók 1D, 2D és 3D elrendezéseket támogatnak f16, f32 és i64 típusokhoz.

| Típus burkoló | Alapul szolgáló Futhark típus | Dimenziók | Felhasználás |
| :--- | :--- | :--- | :--- |
| FutharkArray1DF16 | struct_futhark_f16_1d | [len] | Eltolások, 1D vektorok |
| FutharkArray2DF16 | struct_futhark_f16_2d | [sorok][oszlopok] | Súlyok, bemeneti kötegek |
| FutharkArray3DF16 | struct_futhark_f16_3d | [b][s][d] | Kötegelt szekvencia adatok |
| FutharkArray1DI64 | struct_futhark_i64_1d | [len] | Permutációs indexek |

Rögzített memória a gyors átvitelekhez

A gazdagép-eszköz (H2D) és eszköz-gazdagép (D2H) sávszélesség optimalizálásához a PinnedMemory struktúra cudaHostAlloc-ot alkalmaz. Ez oldalzárolt memóriát allokál, amely lehetővé teszi a GPU számára a Közvetlen Memória Hozzáférés (DMA) alkalmazását cudaMemcpy-n keresztül, megkerülve a szabványos CPU memória előkészítési területet.

TrainingStep folyamat

A tanítási folyamat Futhark belépési pontok sorozataként van megvalósítva, amelyek kezelik az előre irányuló menetet, a veszteség számítást és a visszafordítható visszafelé irányuló menetet.

1. Köteg előre irányuló menet

A batch_forward belépési pont 3D bemeneti tenzort dolgoz fel. Minden mintához végrehajtja az rsf_forward-ot, amely a bemenetet két félre osztja (x1, x2) és affin csatolást alkalmaz:

- Skála: y1 = x1 ⊙ exp(clamp(Ws x2 + bs))
- Fordítás: y2 = x2 + (Wt y1 + bt)

2. Veszteség és visszafelé irányuló menet

A batch_compute_loss függvény Átlagos Négyzetes Hibát (MSE) számít f32 pontossággal a stabilitáshoz. Az rsf_backward belépési pont végrehajtja a gradiens számítást. Mivel az RSF réteg visszafordítható, a visszafelé irányuló menet rekonstruálhatja a gradienseket a közbenső aktivációk tárolása nélkül, jelentősen csökkentve a memória terhelést.

3. SFD súly frissítés

A súlyok a Spektrális Fisher Diagonalizáló (SFD) logika segítségével frissülnek. Az sfd_update_mat függvény impulzus alapú frissítéseket valósít meg a kiterjesztett [dim × (dim+1)] súlymátrixokon (az eltolás az utolsó oszlopban van beolvasztva, így külön eltolás-frissítésre nincs szükség). A WeightKind enum azonosítja, hogy melyik paraméter készlet kerül frissítésre (Skála súlyok, Fordítás súlyok vagy a megfelelő sebességek).

GPU műveletek és CUDA kötések

Mátrix műveletek

A Futhark kernel matmul_tiled biztosítja az RSF transzformációk alapját. Csempézett megközelítést alkalmaz, ahol az A mátrix sorai és a B mátrix oszlopai redukálódnak a kimenet előállításához. Kötegelt szekvenciákhoz a batched_matmul leképezi ezt a műveletet a köteg dimenzión.

EmbeddingAccelerator

Az EmbeddingAccelerator (az AccelInterface-ben hivatkozott) kezeli a keresési és gradiens szórt összeadási műveleteket a tanult beágyazásokhoz a GPU-n. Ez integrálódik a trainingStep-be az end-to-end gyorsítás lehetővé tételéhez a token azonosítóktól a frissített súlyokig.

Nyers CUDA interfész

A Futhark által nem lefedett műveletekhez (mint a specifikus memóriakezelés vagy alacsony szintű szinkronizálás) a JAIDE közvetlen C-ABI kötéseket biztosít a CUDA futtatókörnyezethez:

- cudaMalloc / cudaFree: Eszköz memória allokáció.
- cudaMemcpy: Szinkron adatátvitel.
- cudaStreamSynchronize: Finomabb vezérlés a végrehajtási sorok felett.

---

7.2 FRACTALLPU ÉS R-GPU

A JAIDE hardver gyorsítási rétege speciális feldolgozó egységeket biztosít a Mag Relációs Réteghez, amelyeket kifejezetten az Önhasonló Relációs Gráf (SSRG) nem-euklideszi, önhasonló természetének kezelésére terveztek. A FractalLPU (Fraktál Lineáris Feldolgozó Egység) hierarchikus munkaterhelés egyensúlyozást és memória csempézést kezel a gráf topológia alapján, míg az R-GPU (Relációs Gráf Feldolgozó Egység) egy Network-on-Chip (NoC) architektúra aszinkron szimulációját biztosítja, amelyet a gráf izomorfizmusra és a relációs adatfolyamra optimalizáltak.

FractalLPU: Fraktál Lineáris Feldolgozó Egység

A FractalLPU egy csempe alapú gyorsítási architektúra, amely a gráf csomópontokat fizikai számítási egységekre képezi le fraktál dimenziók alapján. FractalTile objektumok hierarchikus struktúráját alkalmazza a memória és számítási erőforrások kezeléséhez.

Munkaterhelés egyensúlyozás és csempézés

A rendszer FractalDimensionConfig-ot alkalmaz a gráf felosztásának meghatározásához. Főbb paraméterek a hausdorff_dim és a box_counting_levels, amelyek meghatározzák a memória csempék rekurzív felosztását.

- FractalTile: Az LPU alapvető egysége. Minden csempe ComputeUnit objektumok készletét tartalmazza és négy gyermekre osztható.
- Terheléselosztás: A balanceLoad függvény biztosítja, hogy egyetlen ComputeUnit se legyen túlterhelve a pending_ops korlátozásával a load_balance_factor alapján.
- Rögzített pontos végrehajtás: Az LPU skálázott egész aritmetikát hajt végre az executeFixedPoint segítségével, coherence tényezőt alkalmazva a bemeneti jelekre a jel csillapítás szimulálásához a fraktál hierarchián keresztül.

FractalLPU leképezési logika

| Komponens | Felelősség | Kód hivatkozás |
| :--- | :--- | :--- |
| mapSSRGNode | Egy hash-elt SSRG csomópontot egy csempén belüli specifikus ComputeUnit-ra képez le. | src/hw/accel/fractal_lpu.zig:107-113 |
| subdivide | Rekurzívan hoz létre gyermek csempéket, amíg el nem éri a min_tile_size-t vagy a box_counting_levels-t. | src/hw/accel/fractal_lpu.zig:90-105 |
| buildHierarchy | Elindítja a root_tile rekurzív felosztását. | src/hw/accel/fractal_lpu.zig:193-195 |

Relációs Gráf Feldolgozó Egység (R-GPU)

Az R-GPU egy szimulált sokmagos architektúra, amelyet aszinkron üzenetküldésre és relációs műveletekre terveztek. ProcessingCore egységek 2D rácsát szimulálja, amelyek Network-on-Chip (NoC) segítségével kapcsolódnak egymáshoz.

Feldolgozó mag és NoC

Minden ProcessingCore független szereplőként működik saját:

1. Helyi gráffal: A SelfSimilarRelationalGraph egy részhalmazával.
2. Üzenetsorral: NoCMessage csomagokat tárol az aszinkron kommunikációhoz.
3. Állapotgéppel: idle, processing, communicating és power_gated állapotok között vált.

Network-on-Chip szimuláció

A NoC NoCMessage struktúrát alkalmaz a magok közötti kommunikáció megkönnyítéséhez. Az üzenetek típusosak (pl. weight_update, graph_sync, isomorphism_result) és prioritásosak.

- XY-útválasztás: Az üzenetek először az X-tengely, majd az Y-tengely mentén kerülnek irányításra a target_core eléréséhez.
- Teljesítménykapuzás: A PowerGatingController figyeli a mag aktivitást és az üresjáratban lévő magokat power_gated állapotba helyezi az energiafogyasztás csökkentéséhez.

Vektor Feldolgozó Egység (VPU)

Míg a FractalLPU kezeli a gráf szintű csempézést, a VPU biztosítja az alacsony szintű matematikához szükséges SIMD primitíveket. A SimdVector struktúra burkolja a Zig @Vector típusát az ellenőrzött aritmetika biztosításához.

Főbb VPU műveletek:

- SIMD matematika: Támogatja az add, sub, mul és divChecked műveleteket.
- Relációs műveletek: Tartalmaz fma (Fúzionált Szorzás-Összeadás) és dot szorzatokat.
- Normalizálás: normalize és magnitude biztosított a relációs jel vektorok feldolgozásához.

Vektor típus definíciók

A rendszer több vektor szélességet és típust támogat a VectorType-ban definiálva:

| Típus | Sávok | Igazítás | Felhasználási eset |
| :--- | :--- | :--- | :--- |
| f32x8 | 8 | 32 bájt | Szabványos neurális súlyok |
| f64x4 | 4 | 32 bájt | Nagy pontosságú relációs energia |
| i32x8 | 8 | 32 bájt | FractalLPU rögzített pontos műveletek |

---

7.3 RTL HARDVER LEÍRÁSOK

Ez az oldal dokumentálja a Clash-ben (egy Haskell-alapú HDL fordító) megvalósított Register Transfer Level (RTL) hardver modulokat. Ezek a modulok speciális hardver gyorsítást biztosítanak a memória arbitrációhoz, rangsoroláshoz és index kereséshez a JAIDE rendszeren belül.

MemoryArbiter

A MemoryArbiter modul rögzített prioritású arbitrációs Véges Állapotgépet (FSM) valósít meg a megosztott memória erőforráshoz való egyidejű hozzáférés kezeléséhez több hardver kliens számára. Kölcsönösen kizárólagos hozzáférést biztosít a megosztott memória erőforráshoz, miközben kezeli a kérés-válasz ciklusokat.

Implementációs részletek

Az arbitrátor Mealy gépként van megvalósítva az arbiterT átmeneti függvény segítségével. 4 klienst (NumClients) kezel és rögzített 4 ciklusos (ServiceCycles) kiszolgálási ablakot érvényesít megadott kérésenként.

Állapotok és átmenetek

Az arbitrátor két elsődleges állapotban működik az ArbiterState-ben definiálva:

- ArbIdle: Az arbitrátor átvizsgálja a clientReqs-t a findIndex segítségével az első aktív kérés azonosításához. Ha talál, ArbServing-re vált és hozzáférést biztosít a specifikus ClientID4-nek.
- ArbServing: Az arbitrátor fenntartja az aktuális kapcsolatot a ServiceCycles által meghatározott időtartamig. Növeli a belső számlálót, amíg el nem éri a határt, majd visszatér az ArbIdle-hoz.

Adatfolyam és demultiplexálás

Az arbitrátor egyetlen MemRequest-et ad ki a memória vezérlőnek és MemResponse jelek vektorát vissza a klienseknek. A válaszok a filterResp függvény segítségével kerülnek demultiplexálásra, amely biztosítja, hogy a válasz csak a respClient azonosítóval egyező kliensnek legyen látható.

SSI keresési logika állapottáblázat

| Állapot | Átmeneti feltétel | Művelet |
| :--- | :--- | :--- |
| Idle | Just SearchRequest | Átmenet Fetching-re rootAddr segítségével. |
| Fetching | Just TreeNode | Átmenet Comparing-ra vagy rekurzió. |
| Comparing | key == nodeKey | Leállítás SearchResult(found=True) eredménnyel. |
| Comparing | key < nodeKey | Átmenet Fetching(leftChild)-re. |
| Comparing | key > nodeKey | Átmenet Fetching(rightChild)-re. |

RankerCore

A RankerCore egy hardver gyorsító, amelyet a visszakeresés során a szegmensek pontszámainak kiszámítására terveztek. Pozíció-torzított rangsorolási algoritmust valósít meg, amely az eredményeket az eredeti szekvencia pozíciójuk alapján súlyozza.

Pontozási logika

A mag finalScore-t számít a baseScore és egy számított bias kombinálásával.

- Pozíció torzítás: A computePositionBias segítségével számítva, amely reciprok skálázást alkalmaz: positionBiasScale / (position + 1).
- Skálázási tényező: A positionBiasScale 1000-re van rögzítve.

Állapotkezelés

A RankerState nyomon követi a stateCounter-t (az azonos hash-re vonatkozó szekvenciális lekérdezések észleléséhez) és a lastScore-t. Ha egy új RankRequest megegyezik a lastQuery-vel, a belső rang számláló növekszik; egyébként 1-re áll vissza.

SSISearch

Az SSISearch modul egy hardver motor a Szegmentált Szekvencia Index (SSI) fák bejárásához. Nagy sebességű HashKey64 kulcsok keresését végzi a memóriában tárolt fa struktúrán belül.

Keresési FSM

A motor háromállapotú FSM-et valósít meg a SearchState által definiálva:

1. Idle: SearchRequest-re vár, amely tartalmaz egy searchKey-t és egy rootAddr-t.
2. Fetching: Memória kérést ad ki egy TreeNode-hoz egy specifikus NodeAddr32-nél.
3. Comparing: Miután egy TreeNode megérkezik, a checkNode összehasonlítja a searchKey-t a nodeKey-vel. Ezután dönt, hogy leállítja (megtalálva/nem találva) vagy visszatér Fetching-re a leftChild vagy rightChild esetén.

Korlátok és biztonság:

- Max mélység: A rosszul formált fákban való végtelen ciklusok megelőzéséhez a keresés depthExceeded eredménnyel leáll, ha a currentDepth eléri a MaxSearchDepthConfig-ot (64).
- Null mutatók: A motor kifejezetten ellenőrzi a nullAddr-t (0) a gyermekek lekérésének megkísérlése előtt.

---

8 ELOSZTOTT TANÍTÁS

Az elosztott tanítást a JAIDE-ban egy több GPU-s orchestrációs réteg kezeli, amely NCCL-t (NVIDIA Kollektív Kommunikációs Könyvtár) alkalmaz a nagy teljesítményű kommunikációhoz és Futhark-ot a gyorsított kernel végrehajtáshoz. A rendszer egy-rang-per-eszköz modellt követ, ahol több folyamat szinkronizálja a gradienseket és a köteg statisztikákat a nagy léptékű RSF modellek tanításához.

Rendszer architektúra

Az elosztott tanítási verem áthidalja a magas szintű tanítási logikát az alacsony szintű GPU hardver kezeléssel. A DistributedTrainerFuthark orchestrálja a tanítási ciklust, míg a GPUCoordinator kezeli az alapul szolgáló NCCL kommunikátorokat és CUDA streameket.

Elosztott tréner

A DistributedTrainerFuthark a több GPU-s tanítás központi struktúrája. Integrálja az MGT tokenizálót, az RSFAccelerator-t és a Mag Relációs Réteget (NSIR, CREV és ReasoningOrchestrator) egy egységes tanítási interfész biztosításához.

Főbb felelősségek:

- Inicializálás: Rang-tudatos környezetek beállítása, ahol minden tréner példány ismeri a world_size-t és a rank-ot.
- Folyamat végrehajtás: A trainStepFuthark futtatása, amely kezeli a tokenizálást, a beágyazás kereséseket és az előre/visszafelé irányuló meneteket Futhark kerneleken keresztül.
- Relációs integráció: A runCoreRelationalPass periodikus futtatása a neurális frissítések szinkronizálásához a SelfSimilarRelationalGraph-gal.
- Ellenőrzőpont: Verzionált modell állapotok mentése és betöltése a klaszteren keresztül.

GPU Koordinátor és NCCL

A GPUCoordinator alacsony szintű kötéseket biztosít az NVIDIA hardveréhez és kommunikációs primitíveihez. Absztrahálja az NCCL és CUDA stream kezelés komplexitását egy tiszta Zig interfészbe.

Főbb jellemzők:

- Eszköz kezelés: Automatikusan leképezi a rangokat a helyi GPU-kra cudaSetDevice segítségével.
- Kollektív műveletek: Szabványos elosztott primitívek megvalósítása, beleértve az allReduce, broadcast, allGather és reduceScatter műveleteket.
- Szinkronizálás: Egy barrier megvalósítás egy dummy allReduce-on keresztül egy dedikált barrier_buffer-en.
- Memória életciklus: Eszköz memória allokáció (cudaMalloc) és gazdagép-eszköz átvitelek kezelése egy elosztott rang kontextusában.

Felhő telepítés

Míg a JAIDE helyi klasztereken futhat, Modal felhő telepítésre van optimalizálva. Ez lehetővé teszi a DistributedTrainerFuthark gyors skálázását nagy teljesítményű A100/H100 példányokon. A telepítési szkriptek kezelik a Futhark által generált C kód konténerizálását és az NCCL könyvtárak linkelését a felhő környezetben.

---

8.1 ELOSZTOTT TRÉNER

A DistributedTrainerFuthark a JAIDE rendszer több GPU-s tanításának elsődleges orchestrátora. Integrálja a Futhark-gyorsított RSF neurális vermet a Mag Relációs Réteggel, kezelve az adatpárhuzamosságot, a gradiens szinkronizálást NCCL-en keresztül és a nagy léptékű adathalmazok nagy teljesítményű I/O-ját.

1. Inicializálás és konfiguráció

A tréner az initWithConfig segítségével inicializálódik, amely beállítja a szükséges komponenseket mind a neurális, mind a relációs feldolgozáshoz. Szigorú validálást érvényesít a modell dimenziókon (amelyeknek párosnak kell lenniük az RSF csatoló rétegekhez) és a rang/világ méret paramétereken.

Komponens összetétel

A tréner több kritikus alrendszert aggregál:

- MGT szókincs: Angol tokenek és morfológiai dekompozíciós szabályok alapkészletével inicializálva.
- RSFAccelerator: Kezeli a Futhark GPU kontextust és a többrétegű RSF súlyokat.
- Relációs verem: Tartalmazza a CREVPipeline-t, a ChaosCoreKernel-t, a SelfSimilarRelationalGraph-ot és a ReasoningOrchestrator-t.
- GPUCoordinator: Kezeli a rang-specifikus eszköz hozzárendeléseket és az NCCL kollektív műveleteket.

2. Elosztott adathalmaz betöltés

A rendszer rang-tudatos JSONL betöltőt alkalmaz az adatpárhuzamosság megvalósításához. Minden rang kiszámítja az adathalmaz saját szeletét a GPU-k közötti átfedés elkerülése érdekében.

Rang particionálási logika

1. Minta számlálás: A teljes mintaszám a JAIDE_TOTAL_SAMPLES-ből vagy fájl átvizsgálással kerül lekérésre.
2. Index számítás: Minden rang meghatározza a start_valid_index-ét és a samples_per_rank-ját a világ mérete és a saját rang azonosítója alapján.
3. JSONL elemzés: Az extractDatasetText függvény JSON objektumokat elemez, kifejezetten a "text" kulcsot keresve.

Adathalmaz particionálási táblázat

| Paraméter | Leírás |
| :--- | :--- |
| base_per_rank | total_samples / world_size |
| remainder | total_samples % world_size |
| start_valid_index | Az aktuális rang eltolása a globális adathalmazban |

3. Tanítási folyamat

A trainStepFuthark függvény valósítja meg a mag tanítási ciklust, amely áthidalja a természetes nyelvi tokenek és a GPU-gyorsított tenzor műveletek közötti szakadékot.

Adatfolyam: tokenizálás, beágyazás, kernelek

1. Tokenizálás: A bemeneti szöveg token azonosítókká konvertálódik az MGT.tokenize segítségével.
2. Beágyazás keresés: A token azonosítók sűrű vektorokra képeződnek le a LearnedEmbedding rétegben.
3. Futhark előre irányuló menet: A beágyazások FutharkArray2DF16-ként kerülnek feltöltésre a GPU-ra és az RSF rétegeken keresztül kerülnek feldolgozásra.
4. Relációs integráció: A runCoreRelationalPass meghívódik az NSIR gráf és az érvelési állapot frissítéséhez az aktuális köteg alapján.

Gradiens szinkronizálás

Egy korszak vagy köteg szekvencia végén a tréner allReduceFloat32Max-ot (vagy összeget) hajt végre a súly delták szinkronizálásához a klaszteren keresztül.

4. Ellenőrzőpont kezelés

A tréner verzionált ellenőrzőpontokat támogat a tanítás folytonosságának és a modell perzisztenciájának biztosítása érdekében.

Mentés/betöltés mechanizmus

- Verzió követés: A TrainerConfig meghatároz egy checkpoint_version-t (jelenleg v7) a kompatibilitás fenntartásához.
- Szerializáció: Az RSF súlyok, beágyazási mátrixok és az NSIR gráf állapota bináris formátumba kerülnek szerializálva.
- 0-ás rang felelőssége: Általában csak a gyökér rang (0-ás rang) végzi a tényleges fájl I/O-t az ellenőrzőpontokhoz az írási versengés elkerülése érdekében, amelyet egy broadcast követ a többi ranghoz.

---

8.2 GPU KOORDINÁTOR ÉS NCCL

A GPU Koordinátor a JAIDE rendszer több GPU-s elosztott tanításának központi kezelő entitása. Egy-rang-per-eszköz modellt valósít meg, kezelve a CUDA eszközök, memória allokációk és nagy teljesítményű kollektív kommunikációk életciklusát NCCL (NVIDIA Kollektív Kommunikációs Könyvtár) kötéseken keresztül.

Architektúra és eszközkezelés

A GPUCoordinator struktúra kezeli a folyamat rangjának az elosztott "világban" és a hozzárendelt fizikai GPU-nak a kapcsolatát. Biztosítja, hogy minden folyamat egy specifikus CUDA eszközhöz legyen rögzítve a cudaSetDevice segítségével a rangja alapján.

Adatfolyam: Inicializálás

1. Eszköz hozzárendelés: A koordinátor meghatározza a device_id-t a rang és a helyi eszközszám modulójának kiszámításával.
2. NCCL inicializálás: NCCL kommunikátort (ncclComm) inicializál az összes rangon megosztott egyedi azonosító segítségével.
3. Stream létrehozás: Dedikált CUDA stream kerül létrehozásra az aszinkron kollektív műveletek számára a gazdagép oldali végrehajtás blokkolásának elkerülése érdekében.
4. Barrier beállítás: Egy kis 4 bájtos puffer kerül allokálásra az eszközön a barrierek megkönnyítéséhez dummy kollektív műveletek segítségével.

Eszköz memória kezelés

A koordinátor egyszerűsített interfészt biztosít az eszközön tárolt memória kezeléséhez, a nyers CUDA mutatókat Zig-barát absztrakciókba burkolva.

| Függvény | Cél | Implementációs részlet |
| :--- | :--- | :--- |
| allocDeviceMemory | Bájtokat allokál az aktuális GPU-n. | Meghívja az nccl.cudaMalloc-ot. |
| freeDeviceMemory | Felszabadítja a GPU memóriát. | Meghívja az nccl.cudaFree-t. |
| copyToDevice | Adatokat visz át a gazdagépről az eszközre. | cudaMemcpyHostToDevice-t alkalmaz. |
| copyFromDevice | Adatokat visz át az eszközről a gazdagépre. | cudaMemcpyDeviceToHost-ot alkalmaz. |

Kollektív műveletek

Az elosztott tanítás magja az NCCL kollektívákon alapul. A GPUCoordinator ezeket aszinkron műveletekként teszi elérhetővé, amelyek a belső cuda_stream-en hajtódnak végre.

Támogatott kollektívák

- allReduce: Adatokat kombinál az összes rangból egy redukciós operátor (Összeg, Max stb.) segítségével és az eredményt visszaosztja az összes ranghoz.
- broadcast: Puffert másol egy gyökér rangból az összes többi ranghoz.
- allGather: Adatokat gyűjt az összes rangból és az összesített tömböt osztja el az összes ranghoz.
- reduceScatter: Redukciót hajt végre, majd az eredményt szétszórja a rangok között.
- barrier: Egy allReduce végrehajtásával valósul meg a belső barrier_buffer-en. Ez biztosítja, hogy az összes rang elérte ugyanazt a végrehajtási pontot.

NCCL kötések

A rendszer az NCCL megosztott könyvtárral egy vékony Zig burkolón keresztül kommunikál az nccl_bindings.zig fájlban. Ez a fájl definiálja a szükséges C-ABI típusokat és extern függvényeket.

- Eredménykódok: Az ncclResult_t enum leképezi az NCCL visszatérési kódokat, mint az ncclSuccess és az ncclUnhandledCudaError.
- Adattípusok: Zig/Futhark típusokat képez le NCCL típusokra, mint az ncclFloat32 vagy az ncclBfloat16.
- Redukciós operátorok: Definiál olyan műveleteket, mint az ncclSum, ncclProd és ncclMax.

Modal integráció

Felhő léptékű tanításhoz a ModalGPUClient és a kapcsolódó Python szkriptek orchestrálják az elosztott bináris telepítését.

- Erőforrás specifikáció: A tanítási feladatok csúcskategóriás hardverre vannak konfigurálva, kifejezetten B200 vagy B300 GPU-kat kérve.
- Környezet beállítás: A Modal image az nvidia/cuda:12.8.1-devel-ubuntu24.04 alapján épül és tartalmazza a szükséges libnccl2 és libnccl-dev könyvtárakat.
- Feladat telepítés: A deployTrainingJob függvény szerializálja a tanítási paramétereket és elküldi azokat a Modal API-hoz.

---

9 BIZTONSÁG, ELLENŐRZÉS ÉS VÉDELEM

A JAIDE rendszer többrétegű biztonsági és helyességi architektúrát tartalmaz, amelyet a modell következtetés integritásának, a tanítási adatok adatvédelmének és az alapvető algoritmusok matematikai megalapozottságának biztosítására terveztek. Ez az alrendszer áthidalja az alacsony szintű memória biztonsági primitíveket a magas szintű kriptográfiai bizonyítékokkal és formális ellenőrzéssel.

Rendszer biztonsági és védelmi áttekintés

A biztonsági architektúra négy fő területre épül:

- Formális ellenőrzés: oftb.lean és security_proofs.zig
- Nulla-Tudás bizonyítékok: VerifiedInferenceEngine és ZKInferenceProver
- Adathalmaz adatvédelem: HomomorphicEncryption és DatasetFingerprint
- Memória biztonság: safeIntCast és SecureRng

Ellenőrzött Következtetési Motor

A VerifiedInferenceEngine "kötelezd el-majd-bizonyítsd" életciklust biztosít a modell végrehajtáshoz. Biztosítja, hogy az RSF (Visszafordítható Szórt Folyam) verem által generált kimenet egy specifikus bemenet és modell állapot determinisztikus eredménye, anélkül, hogy felfedné a belső súlyokat.

Főbb jellemzők:

- Elkötelezési sémák: Blake3-at alkalmaz a bemeneti/kimeneti elkötelezésekhez.
- Nyom rögzítés: Működési nyomot rögzít a következtetés során a ProofOfCorrectness segítségével.
- Skálázható ellenőrzés: BatchVerifier-t és ProofAggregator-t valósít meg Merkle fák segítségével több következtetés egyidejű ellenőrzéséhez.

Adathalmaz adatvédelem és elhomályosítás

A JAIDE kriptográfiai elhomályosítás és statisztikai adatvédelmi intézkedések kombinációján keresztül védi az érzékeny tanítási adatokat. A HomomorphicEncryption modul a Paillier kriptoszisztémát valósítja meg, lehetővé téve korlátozott aritmetikai műveleteket titkosított adatokon.

Formális ellenőrzés és biztonsági primitívek

A JAIDE megbízhatóságának alapja biztonsági primitívek készlete, amelyek megakadályozzák a szoftver általános sebezhetőségeit, mint az egész szám túlcsordulások és a mutató helytelen igazítása.

Biztonsági segédprogramok:

- safeIntCast: Validálja az előjelet és a bit szélességet az IntegerOverflow és IntegerUnderflow megelőzéséhez.
- safePtrCast: Biztosítja, hogy a mutatók nem null értékűek és helyesen igazítottak a célhoz.

SecureRng

A SecureRng struktúra hibrid megközelítést valósít meg az entrópiához. Az std.crypto.random rendszer által biztosított kriptográfiai véletlenszerűséget keveri egy Lineáris Kongruenciális Generátor (LCG) tartalék állapottal a magas minőségű véletlenszerűség biztosítása érdekében még nagy versengés esetén vagy korlátozott entrópia forrásokkal rendelkező környezetekben is.

Kriptográfiai primitívek

Az érzékeny adatkezeléshez a JAIDE biztosítja:

- secureZeroBytes: Biztosítja, hogy a memória törlésre kerüljön anélkül, hogy a fordító optimalizálná el.
- constantTimeCompare: Megakadályozza az időzítési támadásokat azáltal, hogy bájt puffereket rögzített számú ciklusban hasonlít össze.

Formális ellenőrzés

A JAIDE formális ellenőrzést alkalmaz a legkritikusabb algoritmusok helyességének bizonyítására, kifejezetten a neurális rétegben alkalmazott Ortogonális Fraktál Transzformációs Blokkhoz (OFTB).

Lean4 bizonyítékok (oftb.lean)

Az src/verifaction/oftb.lean fájl Lean4 tételeket tartalmaz, amelyek validálják a split_at művelet tulajdonságait, amely alapvető az OFTB pillangó stílusú keveréséhez.

Bizonyíték motor (formal_verification.zig)

A formal_verification.zig fájl futásidejű bizonyíték ellenőrzőt valósít meg a gráf invariánsokhoz. InvariantType-ot (pl. MEMORY_SAFETY, COHERENCE) és ProofRule-t (pl. MODUS_PONENS, INDUCTION) definiál a SelfSimilarRelationalGraph állapotának validálásához.

Biztonsági tulajdonságok és típuselmélet

A JAIDE biztonsági modellje formális típuselmélet és információáramlás vezérlés alapján épül fel.

BigInt512 aritmetika

A homomorf titkosításhoz és nagy léptékű koordináta rendszerekhez a JAIDE BigInt512 aritmetikát valósít meg a safety.zig fájlban. Ez tartalmaz konstans idejű összehasonlítást és biztonságos nullázást annak biztosítására, hogy a nagy egész szám műveletek ne szivárogtatnak ki oldalsó csatorna információkat.

---

10 TESZTELÉS ÉS BENCHMARKING

A JAIDE kódbázis átfogó teljesítmény benchmark és stressz teszt csomagot tartalmaz, amelyet a mag matematikai és relációs alrendszerek hatékonyságának és helyességének validálására terveztek. Ez az infrastruktúra biztosítja, hogy az optimalizálások - mint a SIMD vektorizáció, a többszálú mátrixszorzás és a lock-free referenciaszámlálás - stabil és teljesítő maradjanak az architektúrális változások során.

Magas szintű teszt architektúra

A tesztelési infrastruktúra három elsődleges kategóriára van osztva:

1. Teljesítmény benchmarkok: Dedikált futtatható fájlok, amelyek mérik az áteresztőképességet (GFLOPS, elemek/mp) a kritikus útvonalakon, mint az RSF és a Tenzor műveletek.
2. Stressz tesztek: Nagy párhuzamossági környezetek, amelyek versenyhelyzeteket keresnek a memóriakezelésben és a referenciaszámlálásban.
3. Egységtesztek: Build rendszerbe integrált tesztek az alrendszerek logikájának validálásához, mint az NSIR és a CREV.

A benchmark csomag egy központi függőségi modulra támaszkodik, az src/_bench_deps.zig-re, amely belső névtereket (rsf, core_tensor, sfd) tesz elérhetővé a tesztelési futtatók számára.

Benchmark csomag

A teljesítmény csomag értékeli a rendszer neurális és matematikai primitíveinek számítási korlátait.

- RSF áteresztőképesség: A bench_rsf méri az elemek-per-másodperc feldolgozást a Visszafordítható Szórt Folyam modell előre és visszafelé irányuló menetei során. Ellenőrzi a verem matematikai invertálhatóságát is.
- Lineáris algebra: A bench_matmul benchmarkol a csempézett, gyorsítótár-barát mátrixszorzást (i-p-j ciklus sorrend) változó mátrix méreteken (128-tól 1024-ig), GFLOPS-ban jelenti a teljesítményt.
- SIMD műveletek: A bench_tensor_ops az elemenként végzett sávszélesség kihasználásra összpontosít a fill, add és mul műveleteknél nagy folytonos memória blokkokra (4M elem), GB/s-ban jelenti az eredményeket.
- Optimalizálási sebesség: A bench_sfd profilálja az FP4 kvantálási logikát és a SpectralNormalizer hatványiterációkat, összehasonlítva a "teljes" és "ritka" frissítési sebességeket.

Stressz és egységtesztek

A stressz tesztelés kritikus a JAIDE egyedi memóriakezeléséhez, különösen a Tensor referenciaszámlálási rendszerhez, amely atomi műveleteket alkalmaz a szálbiztonsághoz.

- Referenciaszámlálás stressz: A stress_tensor_refcount.zig több szálat indít (alapértelmezés 12), amelyek ezernyi véletlenszerű retain és release műveletet hajtanak végre egy megosztott tenzor készleten. A teszt validálja, hogy az összes tenzor végső referenciaszámlálója pontosan 1-re tér vissza, biztosítva, hogy nem történt szivárgás vagy dupla felszabadítás versengés alatt.
- Alrendszer egységtesztek: A build rendszer specifikus teszt célokat definiál:
  - test-tensor: Validálja az alak/lépés logikát és az alapvető matematikát.
  - test-nsir: Biztosítja a gráf topológia integritását és az SHA-256 hashelést.
  - test-crev: Validálja az oksági érvelési lánc kivonást.
  - test-temporal: Ellenőrzi a nanoszekundum pontosságú állapot pillanatképeket.

---

10.1 BENCHMARK CSOMAG

A JAIDE benchmark csomag átfogó teljesítményértékelési eszközöket biztosít a rendszer mag számítási komponenseihez. Ezek a benchmarkok a Visszafordítható Szórt Folyam (RSF) rétegeket, a többszálú tenzor aritmetikai motort, a SIMD-vektorizált elemenként végzett műveleteket és a Spektrális Fisher Diagonalizáló (SFD) optimalizáló primitíveket célozzák.

A csomag különböző munkaterhelések teljesítményének validálására van tervezve, biztosítva, hogy az optimalizálások, mint a csempézett mátrixszorzás és az FP4 kvantálás, megfeleljenek a JAIDE architektúra áteresztőképességi követelményeinek.

Függőség aggregáció

A benchmark csomag centralizált függőségi modult alkalmaz a belső névterek tesztelési futtatók számára való elérhetővé tételéhez.

| Névtér | Forrásfájl | Leírás |
| :--- | :--- | :--- |
| rsf | src/_bench_deps.zig | Visszafordítható Szórt Folyam neurális verem komponensek. |
| core_tensor | src/_bench_deps.zig | Alapvető Tensor műveletek és memória elrendezés. |
| sfd | src/_bench_deps.zig | Spektrális Fisher Diagonalizáló és optimalizálási primitívek. |

RSF áteresztőképesség (bench_rsf)

A bench_rsf modul méri a Visszafordítható Szórt Folyam processzor áteresztőképességét mind előre, mind visszafelé irányban. Mivel az RSF rétegek bijektívek, a visszafelé irányuló menet mind a gradiens propagáláshoz, mind az inverz következtetéshez alkalmazható.

Implementációs részletek

- Konfiguráció: Alapértelmezés szerint 512-es dimenzió, 128 réteg és 64-es köteg méret.
- Folyamat:
  1. Inicializál egy RSF modell példányt.
  2. 20 iterációs bemelegítési fázist hajt végre.
  3. 200 időzített iterációt hajt végre a model.forward(&y) segítségével.
  4. 200 időzített iterációt hajt végre a model.backward(&grad_output, &x, &y, &grad_input) segítségével.
  5. Ellenőrzi az invertálhatóságot a model.verifyInvertible segítségével a numerikus stabilitás biztosításához.

Mátrixszorzás (bench_matmul)

A bench_matmul segédprogram értékeli a csempézett, gyorsítótár-barát i-p-j mátrixszorzás implementáció teljesítményét a Tensor osztályban.

Teljesítmény mérőszámok

A benchmark 128, 256, 512 és 1024 méretű négyzetes mátrixokon iterál. Minden mérethez kiszámítja:

- Teljes idő: Kumulatív idő 100 iterációhoz.
- Iterációnkénti: Átlagos késleltetés matmul hívásonként.
- Áteresztőképesség (GFLOPS): 2.0 × N^3 × iterációk / másodpercek képlettel számítva.

Tenzor elemenként végzett műveletek (bench_tensor_ops)

Ez a benchmark a SIMD-vektorizált elemenként végzett műveletekre összpontosít nagy folytonos memória blokkokra (4M elem). Méri a Tensor implementáció memória sávszélesség kihasználását.

Értékelt műveletek

| Függvény | Leírás |
| :--- | :--- |
| benchFill | Méri a t.fill(val) sebességét és GB/s sávszélességét. |
| benchAdd | Méri az a.add(&b) elemenként végzett összeadást. |
| benchMul | Méri az a.mul(&b) elemenként végzett szorzást. |

SFD optimalizáló primitívek (bench_sfd)

A bench_sfd benchmark a Spektrális Fisher Diagonalizáló által alkalmazott specifikus matematikai kerneleket célozza, kifejezetten az FP4 kvantálást és a Spektrális Normalizálást.

FP4 kvantálás

A benchmark teszteli a quantizeFP4 logikát, amely értékeket vág [-6.0, 6.0] tartományra és diszkrét 4 bites lebegőpontos reprezentációra képezi le azokat. 1M értéket dolgoz fel 100 iteráción keresztül az elemenkénti nanoszekundum meghatározásához.

Spektrális normalizálás

Értékeli a SpectralNormalizer.normalizeWeights függvényt. A benchmark összehasonlítja:

1. Teljes hatványiterációk: 20 iteráció a nagy pontosságú szinguláris érték becsléshez.
2. Ritka hatványiterációk: 5 iteráció a tanítás során végzett gyors közelítéshez.

---

10.2 STRESSZ TESZTEK ÉS EGYSÉGTESZTEK

A JAIDE tesztelési infrastruktúra biztosítja a rendszer matematikai helyességét, memória biztonságát és párhuzamos stabilitását. Ez az oldal részletezi a párhuzamos referenciaszámlálás speciális stressz tesztjeit és a Zig build rendszerben definiált egységtesztek csomagját a mag relációs és neurális komponensekhez.

1. Stressz teszt: stress_tensor_refcount

A stress_tensor_refcount segédprogram dedikált eszköz a Tensor referenciaszámlálási mechanizmus szálbiztonságának validálásához. Mivel a JAIDE Másolás-íráskor (CoW) szemantikára és megosztott memóriára támaszkodik több szálon keresztül (pl. matmul vagy elosztott tanítás során), a retain() és release() atomi integritása kritikus.

Implementációs részletek

A teszt több szálat indít, amelyek egyidejűleg véletlenszerű referencia műveleteket hajtanak végre egy megosztott Tensor objektum készleten.

- Szinkronizálás: Egy std.atomic.Value(usize) barrier biztosítja, hogy az összes szál egyidejűleg kezdje el a műveleteket a versengés maximalizálásához.
- Munkaterhelés: Minden threadWorker konfigurálható számú műveletet hajt végre (ops_per_thread). A műveletek tartalmazzák az egyszeres retain-eket, dupla retain-eket és több tenzoros retain-eket a komplex adatfolyamok szimulálásához.
- Ellenőrzés: Miután az összes szál csatlakozik, a teszt ellenőrzi, hogy minden tenzor végső referenciaszámlálója pontosan 1-re tért vissza (az eredeti tulajdonosi referencia).

Referenciaszámlálás stressz teszt adatfolyam

| Rendszer fogalom | Kód entitás |
| :--- | :--- |
| Párhuzamos munkás | threadWorker |
| Atomi barrier | std.atomic.Value(usize) |
| Referencia növelés | Tensor.retain() |
| Referencia csökkentés | Tensor.release() |
| Biztonsági ellenőrzés | getRefcount |

2. Build rendszer egységtesztek

A JAIDE a Zig build rendszert alkalmazza moduláris teszt lépések definiálásához. Ezek egyenként vagy összesítve futtathatók a test-all lépésen keresztül.

2.1 Mag relációs tesztek

Ezek a tesztek validálják az NSIR (Önhasonló Relációs Gráf) és az érvelési folyamatok integritását.

| Teszt lépés | Célmodul | Validálási hatókör |
| :--- | :--- | :--- |
| test-nsir | nsir_core.zig | Csomópont/él létrehozás, kvantum kapu alkalmazás és topológia hashelés. |
| test-reasoning | reasoning_orchestrator.zig | Energia számítás, állapot pillanatképek és ESSO szimmetria észlelés. |
| test-crev | crev_pipeline.zig | Oksági érvelés, hármas kivonás és validálási láncok. |
| test-temporal | temporal_graph.zig | QuantumState pillanatképek és idősor gráf evolúció. |
| test-surprise | surprise_memory.zig | Jaccard-disszimilaritás szűrés és CAS elkötelezési küszöbök. |

2.2 Neurális és memória tesztek

Ezek validálják az alapvető matematikai és memóriakezelési primitíveket.

- test-tensor: Validálja a Tensor alak/lépés elrendezést, a SIMD-vektorizált elemenként végzett műveleteket és a bináris szerializációs formátumot.
- test-memory: Validálja a speciális allokátorokat, beleértve az ArenaAllocator-t, SlabAllocator-t és BuddyAllocator-t a töredezettség és teljesítmény szempontjából.
- test-rsf: Validálja az RSFLayer affin csatolást (skála S és fordítás T) és az előre/inverz menet visszafordíthatóságát.
- test-oftb: Validálja az Ortogonális Fraktál Transzformációs Blokk pillangó stílusú keverési transzformációit.

3. Teszt végrehajtás és konfiguráció

Tesztek futtatása

A tesztek a zig build paranccsal hajthatók végre. A felhasználók specifikus alrendszereket vagy a teljes csomagot célozhatják:

zig build test-all

zig build test-tensor
zig build test-nsir

zig build test-rsf -Dgpu=true

Optimalizálási statisztikák

A relációs optimalizálási tesztek során (pl. ESSO) a rendszer OptimizationStatistics-t követ nyomon a sztochasztikus folyamatok helyes konvergenciájának biztosítása érdekében.

Optimalizálási mérőszámok követése:

- iterations_completed
- moves_accepted
- best_energy
- temperature

4. Hibakezelés a tesztekben

A teszt csomag szabványosított C-kompatibilis hibakódok készletét alkalmazza a c_api-ban definiálva, biztosítva, hogy a mag relációs réteg meghibásodásai nagy granularitással kerüljenek jelentésre.

| Hibakód | Jelentés |
| :--- | :--- |
| JAIDE_ERROR_ALLOCATION | Memória meghibásodás a speciális allokátorokban. |
| JAIDE_ERROR_NODE_NOT_FOUND | NSIR gráf keresési meghibásodás. |
| JAIDE_ERROR_MATH_ERROR | Túlcsordulás vagy alulcsordulás a neurális/kvantum műveletekben. |
| JAIDE_ERROR_THREADING | Mutex versengés vagy atomi meghibásodás. |

---

11 SZÓJEGYZÉK

Ez az oldal technikai definíciókat és kód-specifikus mutatókat biztosít a JAIDE rendszer architektúrális komponenseihez, matematikai primitívjeihez és kognitív fogalmaihoz.

1. Architektúrális paradigmák

5. gyök architektúra

A JAIDE alapvető paradigmája, amely a Perceptron, CNN, RNN és Transformer után következik. A Visszafordítható Szórt Folyam (RSF) segítségével valósul meg, amely a bijektivitást és az O(dim) memória komplexitást helyezi előtérbe.

RSF (Visszafordítható Szórt Folyam)

Kereszt-affin csatoló rétegekből és determinisztikus szórt permutációkból álló neurális architektúra. Minden réteg bijektív, lehetővé téve az aktivációk pontos inverz rekonstrukcióját a visszafelé irányuló menet során aktiváció gyorsítótár nélkül.

- Implementáció: LayerCore az src/processor/rsf.zig fájlban.
- Matematikai forma:
  - Előre: y1 = x1 ⊙ exp(clip(Ws · x2 + bs))
  - Inverz: x2 = y2 - Wt · y1 - bt

Mag Relációs Réteg

A JAIDE kognitív alrendszere, amely magas szintű érvelést, gráf alapú tudásreprezentációt és kvantum-inspirált optimalizálást kezel.

2. Neurális tér és kód entitás leképezés

Az RSF feldolgozási folyamat:

A felhasználói prompt (karakterlánc) a MorphoGraphTokenizer (mgt.zig) segítségével tokenizálódik, majd a LearnedEmbedding (learned_embedding.zig) beágyazásokat végez, az RSF modell (rsf.zig) feldolgozza, az OFTB (oftb.zig) szórást/gyűjtést végez, majd az inverseInPlace() aktiváció rekonstrukciót hajt végre.

3. Mag terminológia táblázat

| Kifejezés | Definíció | Kód mutató |
| :--- | :--- | :--- |
| NSIR (SSRG) | Önhasonló Relációs Gráf. Egy gráf, ahol az élek kvantum-inspirált korrelációkat képviselnek a tokenek között. | src/core_relational/nsir_core.zig |
| EdgeQuality | Enum, amely meghatározza egy gráf él állapotát: szuperpozíció, összefonódott, koherens, összeomlott vagy fraktál. | src/core_relational/nsir_core.zig |
| OFTB | Ortogonális Fraktál Transzformációs Blokk. Paraméter nélküli Haar-wavelet alapú keverési réteg O(1) memóriával. | src/processor/rsf.zig |
| SFD | Spektrális Fisher Diagonalizáló. Másodrendű optimalizáló, amely Fisher információs mátrix átló becslést alkalmaz. | src/optimizer/sfd.zig |
| SSI | Önhasonló Index. Pozíció-megőrző külső memória struktúra, amely O(log n) visszakeresést tesz lehetővé. | src/index/ssi.zig |
| ESSO | Összefonódott Sztochasztikus Szimmetria Optimalizáló. Gráf topológiát optimalizál szimulált hűtéssel a szimmetriákon. | src/core_relational/reasoning_orchestrator.zig |
| Qubit | Komplex értékű primitív (Complex(f64)), amelyet a csomópont állapotok reprezentálásához alkalmaznak az NSIR gráfban. | src/core_relational/nsir_core.zig |
| ThoughtLevel | Hierarchikus érvelési fázisok: helyi (token szintű), globális (kontextus szintű) és meta (rendszer szintű). | src/core_relational/reasoning_orchestrator.zig |

4. Alrendszer specifikus fogalmak

Memóriakezelési primitívek

- MemoryBlockState: Meghatározza egy memória blokk életciklusát: szabad, allokált, összefonódott vagy migrálódó.
- PinnedMemory: cudaHostAlloc segítségével allokált memória a nagy sebességű gazdagép-eszköz átvitelek megkönnyítéséhez.

Kriptográfia és ellenőrzés

- HomomorphicEncryption: A Paillier kriptoszisztéma implementációja additív homomorf műveletekhez érzékeny adathalmazokon.
- ZKProofBundle: Tároló a Groth16 bizonyítékokhoz, nyilvános jelekhez és ellenőrzési állapothoz a nulla-tudás következtetéshez.

Hardver gyorsítás

- WeightKind: Súlytípusok felsorolása (pl. weights_s, weights_t, velocity_s), amelyeket a Futhark/CUDA gyorsítói interfész alkalmaz.
- FutharkContext: Kezeli a Futhark GPU futtatókörnyezet életciklusát, beleértve az eszköz kiválasztást és a parancs szinkronizálást.
