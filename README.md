# C2P-Net: Zweistufige nicht-starre Punktwolkenregistrierung für die Mittelohrdiagnostik

 ## Installation
 Alle benötigten Packages sind in requirements.txt aufgeführt. <br>
 Außerdem müssen die Installationsanweisungen der sub-repositorys [NgeNet](https://github.com/zhulf0804/NgeNet), [RegTR](https://github.com/yewzijian/RegTR) und [Neural Deformation Pyramid](https://github.com/rabbityl/DeformationPyramid) befolgt werden.

 ## Datenset
 Zum Training und Testen ist eine Datenset notwendig, 
 welches unter mesh_dataset/ear_dataset bereitgestellt werden muss.

```text
 mesh_dataset\
│
├── ear_dataset\
│   ├── 000000\
│   │   ├── intra_surface.stl
│   │   └── pre_surface_with_displ.vtp
│   │
│   ├── 000001\
│   │   ├── intra_surface.stl
│   │   └── pre_surface_with_displ.vtp

...

│   ├── 999999\
│   │   ├── intra_surface.stl
        └── pre_surface_with_displ.vtp
```

Die Dateien *pre_surface_with_displ.vtp* enthalten das source modell, das in Punkten und Verbindungen zwischen diesen dargestellt wird, <br>
und das displacement field (Vektor für jeden Punkt, Transformation von source zu target).
Die Datein *intra_surface.stl* enthält die target 3D-Modelle.

Danach müssen die Daten in einer Pickle-Datei gecached werden.<br>
Dazu einfach diese zwei Zeilen ausführen:<br>

    cd mesh_dataset
    pathon ear_dataset_setup.py

Das Skript erstellt dabei auch automatisch einen Train-Val-Test (80%, 15%, 5%) Split für das Datenset.
Leider ist es zurzeit nicht möglich, diesen Datensatz bereitzustellen, weil zunächst rechtliche Angelegenheiten mit dem Institut geklärt werden müssen.

## Training
Zum Trainieren von *NgeNet*:<br>

    python trainNgeNet.py

Das Ergebnis ist unter *trainResults/eardataset_nonrigid_randSurface_perm* zu finden. <br>

Zum Trainieren von *RegTR*:<br>

    python trainRegTR.py

Außerdem kann TensorBoard mit diesen Daten die Loss Curve usw. visualisieren. Die weights sind im Unterordner */checkpoints* zu finden.

## Testen
Zum Testen der Methode dieses Skript ausführen:

    python test_pipeline.py

Das Skript wird als Ausgabe unter anderem die Chamfer Distance des Ergebnisses und die wall time ausgeben.<br>
Die Ergebnisse werden als STL-Dateien im *test_output_folder* gespeichert. Außerdem finden sich dort andere Ausgaben, die für das Erstellen der Animationen benötigt werden.<br>

Zum Erstellen eines Graphen, der zeigt, wie genau die Ergebnisse der Methode sind verglichen zu der Menge an Punkten in target, muss folgender Befehl ausgeführt werden:

    python surfaceamount_test.py 

## Bilder/Animation erstellen

Für das Erstellen der Bilder in der Präsentation folgendes Skript genutzt:

    python make_animations.py [TEST_INDEX]

Über den TEST_INDEX kann das Testsample ausgewählt werden. Der maximale Index ist die Anzahl an Testsamples -1.

Alle in der BeLL und in der Schriftlichen Dokumentation genutzten Diagramme und Bilder sind unter figures/ zu finden.



### Danksagung
Ich danke **Liu Peng vom NCT Dresden** und **Steffen Oßmann vom ERCD Dresden** für das Bereitstellen des Datensets. <br>
Außerdem danke ich den Erstellern von [NgeNet](https://github.com/zhulf0804/NgeNet), [RegTR](https://github.com/yewzijian/RegTR) und [Neural Deformation Pyramid](https://github.com/rabbityl/DeformationPyramid).