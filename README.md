# PokéGAN

Folgende Schritte müssen durchgeführt werden, um das Projekt auch auf dem eigenen PC zum laufen zu bringen:

1. Clonen Sie das Projekt (oder laden Sie es herunter)
2. Installieren Sie Anaconda (https://www.anaconda.com/products/individual)
3. Öffnen Sie das nun installierte Terminal "Anaconda Prompt"
4. Navigieren Sie im Terminal zu dem gecloneten Projekt
5. Erstellen Sie ein Anaconda environment mit der in dem Verzeichnis liegenden Datei environment.yml mit dem folgenden Befehl (Nötig, um die Dependencies des Projekts herunterzuladen): <br>
conda env create -f environment.yml
6. Aktivieren Sie das zuvor erstellte environment mit folgendem Befehl: <br>
conda activate pokegan
7. Öffnen Sie das Programm "Anaconda Navigator""
8. Öffnen Sie in diesem die App "Jupyter Notebook"
9. Wählen Sie in der Jupyter Notebook App, welche sich in Ihrem Browser geöffnet hat, die Datei "main.ipynb", welche sich im Ordner "src" befindet
10. Nun sollte sich das Projekt geöffnet haben, die nötigen Dependencies sollten aufgelöst werden können und Sie sollten in der Lage sein das Projekt nach Belieben zu verändern und auszuführen
