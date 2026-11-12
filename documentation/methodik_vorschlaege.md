# Vorschläge zur Weiterentwicklung der Analysemethodik

Stand: 10. November 2026. Dieses Dokument sammelt methodische Änderungen am
`sans2d-mode-analyzer`, die aus einem Review des Codes und einer Testanwendung
auf die 10-mT- und 1-T-Daten (`experimental_data/doi_10.48550_arXiv.2606.18791`)
entstanden sind. Für jeden Punkt steht, **warum** die Änderung sinnvoll ist,
**wie** sie sich einbauen lässt und **welcher Vorteil** zu erwarten ist.

Die genannten Zahlen stammen aus Ad-hoc-Skripten außerhalb des Repositorys
(gewichteter Fit, `q_range=(0.015, 0.10)`, 7×7-Subpixel-Gewichte). Sie sind
Plausibilitätschecks, keine Ergebnisse.

> **Ergänzung von Codex, 10. November 2026:** Die nachfolgenden Abschnitte
> „Einschätzung von Codex“ kommentieren die einzelnen Vorschläge; der
> ursprüngliche Vorschlagstext bleibt unverändert. Grundlage sind der Abgleich
> mit Implementierung und Supplement sowie kleine synthetische Gegenprüfungen.
> Die oben genannten experimentellen Zahlen aus externen Ad-hoc-Skripten
> wurden dabei nicht unabhängig reproduziert. Vor einer Umsetzung sollten
> diese Skripte und ihre vollständige Konfiguration ins Repository aufgenommen
> werden.

## Leitidee der Methode

Man fittet eine große, generische Basis und lässt die Daten auswählen, welche
Struktur in diesem Freiraum tatsächlich besetzt ist. Die Überparametrisierung ist
dabei gewollt. Daraus ergeben sich zwei Fragen, an denen sich alle Kennzahlen
messen lassen sollten:

1. **Signalqualität:** Gibt es ungerade Anteile, die dort nicht sein sollten, und
   welche Ursache haben sie?
2. **Struktur:** Welche Winkelordnungen sind stabil und signifikant enthalten, und
   welche fitten nur Rauschen?
Die vorhandene Basis, der WLS-Fit, die Kovarianz, die Blocktests und die
radialen Spektren tragen dieses Konzept bereits. Die Vorschläge betreffen vor
allem die **abgeleiteten Kennzahlen**. Diese beruhen bisher auf Leistung
(ĉ²) statt auf Signifikanz. In einer bewusst überparametrisierten Basis
enthalten sie deshalb einen systematischen Rauschbeitrag, oder ihre Skala hat
keinen sinnvollen Nullpunkt.

Priorität: **1 → 2 → 3**, danach 4 und 5.

---

## 1. Strukturauswahl über Signifikanz statt über Leistung

### Warum

`power_model_selector` und die 95-%-Rekonstruktion ordnen die Moden nach ĉ².
Damit beantworten sie die Frage „Was ist stabil enthalten?“ nicht:

- Verrauschte Moden mit großer Varianz werden genauso aufgenommen wie gut
  bestimmte.
- Da die Leistung ∫I² q dq vom kleinen q dominiert wird, gewinnt praktisch immer
  der isotrope Anteil bei kleinem q. Bei 10 mT besteht die 95-%-Rekonstruktion
  aus 5 Moden, alle mit n = 0. Durch die Trunkierung der radialen Moden schwingt
  sie nach und wird bei q ≳ 0.08 negativ.
- Bei 1 T sind für 95 % der Leistung 643 Moden nötig. Dort misst die Leistung
  fast nur Rauschen (siehe Punkt 2).

Die Blocktests `angular_block_tests` beantworten die Frage dagegen direkt. Es
fehlt nur eine Korrektur für das Mehrfachtesten: Bei 51 Blöcken und 95 %
Konfidenz sind ohne Signal etwa 2–3 falsch positive Treffer zu erwarten.

### Wie

Eine neue Funktion, zum Beispiel `significant_structure(result, alpha=0.05,
correction="holm")` in einem neuen Modul `structure.py`:

1. p-Werte aller existierenden Blöcke (cos n, sin n) aus `result.block_tests`
   einsammeln.
2. Holm-Bonferroni anwenden (kontrolliert die familienweise Fehlerrate).
   Alternativ Benjamini-Hochberg, wenn eine FDR-Kontrolle gewünscht ist.
3. Die Moden aller signifikanten Blöcke mit allen radialen Ordnungen m behalten
   und das reduzierte Modell **neu fitten**. Dafür auf den Originaldaten
   `fit_wls` bzw. bei Pixelintegration `fit_design_matrix` mit den
   entsprechenden Spalten verwenden. Die Koeffizienten des großen Fits sollten
   nicht einfach abgeschnitten werden.
4. Verschachtelter Modellvergleich reduziert gegen voll:
   - bei absoluten Unsicherheiten Δχ² = χ²_red − χ²_voll gegen χ²_{K−K_red};
   - bei aus den Residuen geschätzter Skala
     F = (Δχ²/(K−K_red)) / (χ²_voll/ν) gegen F_{K−K_red, ν}.
5. Rückgabe: ausgewählte Blöcke, korrigierte p-Werte, der reduzierte Fit, sein
   Bild und seine Spektren sowie das Ergebnis des Modellvergleichs.

Im `standard_analyzer` ersetzt diese Rekonstruktion die 95-%-Zeile im Plot. Die
Leistungsauswahl kann als beschreibende Zusatzinformation erhalten bleiben. Im
Panel der Blocktests sollte zusätzlich die Holm-Schwelle eingezeichnet werden.

### Vorteil

- Ein statistisches statt eines willkürlichen Kriteriums. Ein Prozentsatz der
  Leistung ist für verrauschte Daten bedeutungslos.
- Interpretierbares Ergebnis. Bei 10 mT bleiben mit Bonferroni genau
  **cos 0, cos 2 und sin 1** übrig: ein isotroper Hauptanteil, eine schwache
  cos-2θ-Anisotropie und ein Zentrierungsartefakt (siehe Punkt 3), sonst Rauschen.
- Der F-Test belegt, dass die verworfene Struktur tatsächlich mit Rauschen
  verträglich ist.

### Tests

- Synthetisches Muster aus cos 0 und cos 2 plus Rauschen: Es werden genau diese
  Blöcke gewählt, und der F-Test ist unauffällig.
- Reines Rauschen: Im Mittel weniger als α · (Zahl der Blöcke) falsch positive
  Treffer, mit Holm höchstens selten einer.

### Einschätzung von Codex

**Sinnvolle Ergänzung, mit einer wichtigen Einschränkung beim Modellvergleich.**
Holm-korrigierte Blocktests und ein anschließender Refit sind empfehlenswert.
Durch Maske, Gewichte und diskrete Abtastung sind die Designspalten im
Datenraum im Allgemeinen nicht orthogonal; das bloße Abschneiden der
Koeffizienten ersetzt den Refit daher nicht.

Die Leistungsauswahl sollte als beschreibende Größe erhalten bleiben: Eine
95-%-Rekonstruktion beschreibt eine normkontrollierte Kompression der
gefitteten Funktion. Sie prüft weder Signifikanz noch Stabilität, ist aber
auch bei verrauschten Daten nicht bedeutungslos. Eine Holm-korrigierte
Signifikanzentscheidung belegt ihrerseits noch keine Stabilität gegenüber
Basisgröße oder q-Bereich.

Beim vorgeschlagenen Schritt 4 wird das reduzierte Modell aus denselben Daten
ausgewählt, an denen es anschließend getestet wird. Die gewöhnliche χ²- bzw.
F-Referenzverteilung ist nach dieser Auswahl nicht automatisch gültig; die
Holm-Korrektur der vorherigen Blocktests behebt dieses Problem nicht. Für eine
belastbare Aussage braucht es vorab festgelegte Modellvergleiche, unabhängige
Validierungsdaten oder eine Kalibrierung des gesamten Auswahl- und
Refit-Verfahrens unter einer klar definierten Nullhypothese. Ein nicht
signifikanter Unterschied beweist zudem nicht, dass verworfene Struktur
Rauschen ist; die Teststärke kann dafür zu gering sein. Siehe
[Berk et al., *Valid post-selection inference*](https://arxiv.org/abs/1306.1059).

Holm kontrolliert die familienweise Fehlerrate auch bei abhängigen Tests,
sofern deren einzelne p-Werte gültig sind. Benjamini-Hochberg benötigt
zusätzliche Abhängigkeitsannahmen; korrelierte Moden erfüllen diese nicht
automatisch. Die Unterschiede sind in der
[R-Dokumentation zu p-Wert-Korrekturen](https://stat.ethz.ch/R-manual/R-devel/library/stats/html/p.adjust.html)
beschrieben. Für synthetische Tests sollten Fehlerraten und Teststärke über
mehrere Rauschrealisierungen geprüft werden. Genau dieselben Blöcke in jeder
Realisierung zu verlangen, wäre kein angemessenes statistisches Testziel.

#### Implementierungsstand Holm-Korrektur (Codex, 10. November 2026)

Die Holm-Korrektur bleibt als bei Bedarf abrufbare numerische Zusatzinformation
in den Blocktest-Ergebnissen verfügbar. Der Standardplot hebt keine
Holm-Entscheidungen mehr hervor. Eine gemeinsame Testfamilie umfasst alle
Cosinus- und Sinus-Blöcke mit positivem numerischem Rang, einschließlich
des isotropen Blocks. Fehlende und Rang-null-Blöcke zählen nicht mit; im
Standardrechteck mit n_max=25 ergeben sich normalerweise 51 Tests.

Die zusätzlichen Ergebnisattribute heißen cosine_holm_pvalue,
sine_holm_pvalue, cosine_holm_reject, sine_holm_reject und holm_family_size.
Sie werden aus den bisherigen Feldern abgeleitet; Funktionsargumente,
Ergebniskonstruktoren und rohe p-Werte bleiben erhalten. Die familienweite
Fehlerrate folgt dem bisherigen Konfidenzparameter:
α = 1 − block_test_confidence, standardmäßig 0.05.

Im T_n-Plot bleibt die gestrichelte Linie ausdrücklich eine nominelle,
modellbedingte Referenz. Sie kennzeichnet keine empirische Artefaktgrenze
und keine eindeutige Identifikation physikalischer Beiträge. Die
vorgeschlagene Strukturauswahl mit reduziertem Refit ist weiterhin ein
eigener methodischer Schritt.

Die Tests umfassen eine unabhängige Gegenrechnung über alle geschlossenen
Bonferroni-Schnittmengentests, unterschiedliche Blockränge, fehlende und
undefinierte Wahrscheinlichkeiten, Bindungen, Grenzwerte und die tatsächlichen
Plotdarstellung. In je 4096 synthetischen Rauschrealisierungen mit abhängigen Blöcken
beträgt die beobachtete familienweite Fehlalarmrate etwa 4.1–4.3 % bei
vorgegebenen 5 %, sowohl unter der vollständigen Nullhypothese als auch bei
einem zusätzlichen echten Signal und für die χ²-/F-Referenzen. Die
unkorrigierten Entscheidungen liegen in denselben Prüffällen bei 28–31 %.
Holm setzt weiterhin gültige einzelne p-Werte voraus und behebt kein
unzutreffendes Kovarianzmodell oder eine datengesteuerte Modellauswahl.
Die simulierte Fehlalarmrate bezieht sich auf das fälschliche Verwerfen
tatsächlich verschwindender Koeffizientenblöcke im jeweiligen Erzeugungsmodell.
Sie ist keine empirisch bestimmte Fehlerrate der Messdaten und keine Rate
physikalischer Fehlinterpretationen. Die Erfüllung der Testvoraussetzungen
für die experimentellen Daten wurde damit nicht nachgewiesen.

Die gesamte Testsuite liefert nach dieser Ergänzung **141 bestandene Tests
und vier bekannte erwartete Fehler** zum relativen geometrischen
Kovarianzmodell. Die bestehenden Numerik- und Plotwarnungen wurden nicht
unterdrückt. Die korrigierten Wahrscheinlichkeiten stimmen
mit der unabhängigen bisherigen Holm-Auswertung der 51 Block-p-Werte
überein.

#### Deskriptive Interpretation des ungeraden Untergrunds (Codex)

Für ein erwartungsgemäß paritätsgerades Observable kann ein breiter Teppich
ungerader Leistungen als empirisches Vergleichsniveau für Messrauschen,
Asymmetrien und Verarbeitungsartefakte dienen. Gerade Leistungen auf oder
unter diesem Niveau sind im Leistungsvergleich nicht belastbar über dem
empirischen Untergrund aufgelöst. Das entspricht einem Kriterium für
Robustheit gegenüber dem sichtbaren Störniveau, keinem kalibrierten p-Wert.
Die bisherige gemeinsame Normierung der Leistungsplots bleibt erhalten.

Eine allgemeine Signifikanzschwelle folgt daraus nicht automatisch:
Modevarianzen, radiale Profile und Empfindlichkeiten können sich unterscheiden.
Die Verwendung ungerader Leistungen als Untergrund setzt außerdem voraus,
dass physikalische ungerade Beiträge für das jeweilige Observable
ausgeschlossen werden können. Die 100-mT- und 1-T-Dateien enthalten nur
einen Spin-Flip-Kanal; eine chirale ungerade Komponente darf dort nicht
automatisch als Rauschen behandelt werden.

Die Holm-Hervorhebung wurde auf Grundlage dieser Interpretationsdiskussion
aus dem Standardplot entfernt. Die mathematische Zusatzrechnung bleibt
separat abrufbar. Es wird keine automatische Schwelle aus den ungeraden
Leistungen abgeleitet.
Die 34 von der Darstellungs- und Supplementanpassung betroffenen Tests
bestehen; die sechs experimentellen Standardgrafiken wurden ohne
Holm-Hervorhebung neu erzeugt.

---

## 2. Rauschkorrektur quadratischer Größen

### Warum

Für einen Block mit Kovarianz C_n gilt E[ĉ_nᵀĉ_n] = c_nᵀc_n + tr(C_n). Alle
Leistungen, Fraktionen und die Paritätsmaße sind daher positiv verzerrt. In
einer großen Basis summiert sich das:

| Datensatz, Basis (25,25) | tr(C)/P (erwartete Rauschleistung relativ zur gefitteten Leistung) |
|---|---|
| 10 mT | 0.011 |
| 1 T   | **3.95** |

Bei 1 T übersteigt die erwartete Rauschleistung die gemessene Gesamtleistung.
Das reduzierte χ² beträgt 0.65, die Fehler sind also vermutlich überschätzt.
Die Leistungsplots zeigen dort im Wesentlichen das Rauschen.

Bei der ungeraden Fraktion ist der Effekt besonders wichtig, weil die Hälfte
aller Moden ungerade ist. Bei 10 mT beträgt die gemessene Fraktion
f_odd = 0.0022, allein aus Rauschen erwartet man 0.0066. Aus f_odd selbst lässt
sich daher nichts über die Signalqualität ablesen.

### Wie

- `angular_mode_powers`: zusätzliche Felder `cosine_noise` / `sine_noise`
  = tr(C_n^b) und `cosine_debiased` / `sine_debiased` = ĉᵀĉ − tr(C). Die
  korrigierten Werte dürfen negativ sein und sollten nicht abgeschnitten werden.
- `parity_power_fractions`: zusätzlich `expected_noise_odd_fraction`
  = tr(C_odd)/P.
- Plot: das Rauschniveau tr(C_n)/M als gestrichelte Linie oder graues Band in den
  Leistungspanels. Was darunter liegt, ist vom Rauschen nicht zu unterscheiden.
- Die Unsicherheit der Leistung nach `eq:power-error` im Supplement bleibt gültig.

### Vorteil

Die Leistungsplots werden lesbar: Man sieht sofort, welche Ordnungen über dem
Rauschen liegen. Die Paritätsfraktion bekommt einen Vergleichswert. Der Aufwand
ist gering (eine Spur pro Block), und es kommen nur Felder hinzu.

### Einschätzung von Codex

**Sehr nützlich als zusätzliche Rauschdiagnostik; mehrere Interpretationen
müssen präzisiert werden.** Für einen unverzerrten Koeffizientenschätzer gilt

$$
\mathbb{E}\!\left[\widehat{\mathbf c}^{T}\widehat{\mathbf c}\right]
= \mathbf c^{T}\mathbf c + \operatorname{tr} C_{\mathrm{sampling}}.
$$

Entscheidend ist die tatsächliche Stichprobenkovarianz. Bei absoluten
Messunsicherheiten wird diese inzwischen auch für zusätzliche geometrische
Pixelgewichte berechnet (Implementierungsstand unten). Der relative,
residualskalierte Modus bleibt an sein Präzisionsmodell gebunden. Die
Unterscheidung und die allgemeine Sandwich-Formel stehen im
[Supplement](supplementary_methods.tex) bei den Gleichungen mit den Labels
**eq:covariance** und **eq:sandwich**. Vor einer statistischen Deutung der
Spurkorrektur muss weiterhin klar sein, welches Rauschmodell verwendet wird.

- Rohleistungen und korrigierte Leistungen sollten gemeinsam erhalten bleiben.
  Negative korrigierte Werte sind zulässig; Abschneiden bei null würde erneut
  verzerren.
- Die positive Verzerrung gilt für die quadratischen Leistungen. Daraus folgt
  keine allgemein positive Verzerrung ihrer Quotienten: Rauschen in geraden
  Moden kann eine tatsächlich große ungerade Fraktion auch verkleinern.
  Insbesondere ist tr(C_odd)/P̂ wegen des zufälligen Nenners nicht
  E[f_odd]. Die Größe sollte etwa „Rauschleistung relativ zur gefitteten
  Gesamtleistung“ heißen; Werte über eins sind möglich.
- Die Spur ist ein **mittlerer Rauschbeitrag, keine Signifikanzschwelle**.
  Beispielsweise ergeben ĉ = (0, 0.1) und C = diag(100, 10⁻⁴) die Leistung
  0.01 < tr(C), aber T = 100: Ein gut bestimmtes Signal in einer Richtung
  kann trotz großer Rauschleistung in einer anderen Richtung hochsignifikant
  sein. Aussagen wie „darunter nicht vom Rauschen unterscheidbar“ sollten
  deshalb nicht als Testentscheidung übernommen werden.
- Eine unveränderte Leistungsunsicherheit nach Abzug der Spur ist nur bei
  festem, bekanntem C unmittelbar gerechtfertigt. Bei einer aus Residuen
  geschätzten Kovarianz ist auch die subtrahierte Spur zufällig; ihre
  Unsicherheit und mögliche Kovarianzen müssen berücksichtigt werden.
  Die Gleichung **eq:power-error** ist bereits für die Rohleistung eine mit
  ĉ ausgewertete Gaußsche Plug-in-Schätzung.
- Gewöhnliche Leistungsfraktionen aus korrigierten Leistungen sind bei einer
  nicht positiven korrigierten Gesamtleistung nicht sinnvoll.
  Ein reduziertes χ² unter eins allein belegt außerdem keine überschätzten
  Messfehler; auch Gewichtung und Korrelationen können die Interpretation
  verändern.

#### Ausgangsprüfung vor der Kovarianzkorrektur (Codex, 10. November 2026)

**Historischer Prüfstand:** Dieser Abschnitt dokumentiert die nachgewiesene
Abweichung vor der Korrektur. Der aktuelle Implementierungsstand folgt unten.

Die Prüffälle stehen jetzt in
[tests/test_covariance_calibration.py](../tests/test_covariance_calibration.py).
Sie ändern weder die API noch die produktive Kovarianzberechnung. Pro
Rauschmodell werden 4096 Realisierungen durch die öffentlichen Fit- und
Blocktest-Funktionen ausgewertet; sowohl der koordinatenbasierte Fit als auch
der Einstieg mit vorberechneter Designmatrix werden geprüft.

**Aufbau:** 48 feste Messpunkte mit unvollständiger Winkelabdeckung,
fünf Koeffizienten (ein nicht verschwindender isotroper Anteil sowie jeweils
zwei unter der Nullhypothese verschwindende cos-2- und sin-1-Koeffizienten).
Das Signal liegt exakt im Modellraum. Die Messunsicherheiten variieren über
die Messpunkte. Die gewichteten Designmatrizen sind gut konditioniert
(Konditionszahl kleiner als 10), sodass die Abweichungen nicht durch eine
instabile Inversion erklärt werden. Für die relative Skalierung beträgt die wahre, dem Fit
unbekannte Standardabweichung das 2.4-Fache des vorgegebenen Maßstabs.
Geometrie und Rauschen haben feste Seeds (913 bzw. 272).

Drei Interpretationen werden getrennt geprüft:

- **Messunsicherheiten:** w = 1, tatsächliche Varianz σ_p².
- **Präzisionsgewichte:** zusätzliche w_p, tatsächliche Varianz σ_p²/w_p.
- **Geometrische Gewichte:** dieselben zusätzlichen w_p, aber tatsächliche
  Varianz weiterhin σ_p². Die Gewichte ändern die Zielfunktion, nicht die
  zugrunde liegende Messvarianz.

Zusätzlich zur Kovarianz werden die Abdeckung der 95-%-Koeffizientenintervalle
und die Null-p-Werte an mehreren Quantilen geprüft. Die statistischen
Toleranzen richten sich nach der Monte-Carlo-Präzision (fünf Standardfehler);
sie verlangen keine exakt 5 % in einer endlichen Stichprobe.

Die beobachteten Fehlalarmraten der bisherigen T_n-Auswertung bei nominell
5 % sind:

| Rauschmodell | Kovarianzskala | cos 2 | sin 1 |
|---|---|---:|---:|
| Messunsicherheiten | absolut | 5.0 % | 6.1 % |
| Messunsicherheiten | aus Residuen | 5.2 % | 5.7 % |
| Präzisionsgewichte | absolut | 4.7 % | 5.8 % |
| Präzisionsgewichte | aus Residuen | 5.0 % | 6.0 % |
| Geometrische Gewichte | absolut | 0.5 % | 0.0 % |
| Geometrische Gewichte | aus Residuen | 23.7 % | 11.5 % |

Die beiden öffentlichen Fit-Einstiege liefern in diesen Prüffällen dieselben
Raten. Bei Residualskalierung ist die wahre gemeinsame Rauschskala wie oben
beschrieben größer; die korrekten Präzisionsmodelle berücksichtigen das.

**Befund:** Die aktuelle Kovarianz und die χ²-/F-Blocktests sind in den
geprüften Fällen mit passenden Präzisionsgewichten kalibriert. Bei
geometrischen Gewichten beschreibt die inverse gewichtete Normalmatrix
dagegen nicht die tatsächliche Koeffizientenstreuung. Der Abgleich mit der
unabhängig per QR berechneten Fortpflanzung des bekannten Messrauschens
bestätigt die allgemeine Stichprobenkovarianz aus dem Supplement
(**eq:sandwich**). Mit dieser bekannten Kovarianz ergeben sich auf denselben
geometrisch gewichteten Fits kalibrierte χ²-Blocktests (etwa 4.7 % bzw.
6.1 % Fehlalarme). Dieser Referenztest verwendet die wahre Rauschskala und
belegt keine allgemeine F-Kalibrierung nach Residualskalierung.

Ein weiterer deterministischer Prüffall verwendet die tatsächlichen
7×7-Subpixel-Flächengewichte und die pixelintegrierte Designmatrix des
Repositories. Auch dort unterscheiden sich Arbeits- und
Stichprobenkovarianz. Separate Nullsimulationen prüfen außerdem korrelierte
und singuläre Koeffizientenblöcke mit dem jeweils beibehaltenen Rang, sowohl
für χ² als auch für eine unabhängig geschätzte χ²-verteilte Skala im
F-Test. Das setzt die bereits korrigierte gemeinsame Toleranz für Rang und
Pseudoinverse statistisch auf die Probe.

**Konsequenz für die Implementierung:** Für eine statistische Auswertung
geometrisch gewichteter Fits muss die Kovarianzberechnung angepasst oder die
Verwendung der bisherigen Arbeitskovarianz ausdrücklich eingeschränkt werden.
Bei bekannten unabhängigen Messunsicherheiten ist die Fortpflanzung durch
die tatsächliche Fit-Abbildung ausreichend. Bei unbekannter gemeinsamer
Skala müssen zusätzlich deren Schätzung und die Referenzverteilung der
Blocktests geklärt werden; eine einfache Residualskalierung behebt das
Problem nicht. Die Bedeutung von Präzisions- und Geometriegewichten darf
dabei nicht stillschweigend gleichgesetzt werden.

Die neun Prüffälle, die die derzeit fehlende statistische Kalibrierung
geometrischer Gewichte verlangen, sind ausdrücklich als **strict xfail**
markiert: vier Kovarianzprüfungen, vier Blocktest-Kalibrierungen und ein
Pixelintegrationsfall. Sie bleiben sichtbar und müssen bei einer späteren
Korrektur überprüft werden; ein unerwartetes Bestehen wird zum Testfehler.
Diese Markierungen gelten für die ausdrücklich geometrische Interpretation
und legen keine neue Standardbedeutung aller sample_weight-Werte fest.
Die Simulationen prüfen unabhängiges Gaußsches Rauschen bei festem Design.
Aus Daten geschätzte Zentren, Interpolationskorrelationen und
Modellabweichungen sind durch diese Prüffälle noch nicht abgedeckt.

Reproduzierbarer gezielter Lauf in einer Umgebung mit den Testabhängigkeiten:

~~~sh
env PYTHONPATH=src OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 \
  python -m pytest tests/test_covariance_calibration.py -o addopts='' -q -s -rx -W error
~~~

**Ausgeführte Validierung:** Die neue Datei liefert 36 bestandene Tests und
neun erwartete Fehler, auch mit Warnungen als Fehler. Zusammen mit
test_basis_and_fit.py, test_block_test.py und test_supplementary_equations.py
ergeben sich 82 bestandene Tests und dieselben neun erwarteten Fehler.
Im gemeinsamen Lauf traten in bestehenden Tests 60 NumPy-Laufzeitwarnungen
bei Matrixmultiplikationen auf (lokale NumPy-2.2.6-/Accelerate-Konfiguration);
die numerischen Vergleichsprüfungen bestanden. Diese Warnungen wurden nicht
unterdrückt oder durch Änderungen an bestehenden Tests umgangen. Die neuen
Prüffälle bestehen den separaten Lauf mit **-W error** ohne solche Warnungen.

#### Implementierte Kovarianzkorrektur (Codex, 10. November 2026)

Für **absolute Pixelunsicherheiten** ist die Korrektur jetzt im gemeinsamen
Fit-Löser implementiert. Mit A als linearer Abbildung der Messdaten auf die
Koeffizienten wird C = A diag(σ²) Aᵀ berechnet. Eine QR-Faktorisierung und
dreieckige lineare Gleichungssysteme vermeiden die Inversion der
Normalmatrix und eine N×N-Messkovarianz. Zusätzliche Flächengewichte ändern
die Fit-Zielfunktion, aber nicht die vorgegebenen Pixelvarianzen.

Die öffentlichen Funktionssignaturen und Ergebnisfelder bleiben erhalten.
Die Berechnung der Koeffizienten und damit die gefitteten Intensitäten,
Rohleistungen und radialen Spektren ist unverändert. Ihre Unsicherheiten und
die T_n-Statistiken verwenden automatisch die fortgepflanzte Kovarianz.
Das gilt sowohl für externe Flächengewichte als auch für die interne
Pixelintegration. Die bestehenden experimentellen Skripte verwenden bereits
absolute Unsicherheiten und benötigen keine geänderten Aufrufe.

Im geometrisch gewichteten synthetischen Prüffall ergeben sich jetzt etwa
**4.7 % (cos 2) und 6.1 % (sin 1)** Fehlalarme bei nominell 5 %; beide liegen
innerhalb der festgelegten Monte-Carlo-Toleranzen. Die empirische
Koeffizientenkovarianz und die 95-%-Intervalle stimmen ebenfalls mit dem
erzeugten Messrauschen überein. Der Pixelintegrationsfall und eine zusätzliche
Invarianzprüfung gegen eine globale Skalierung der Flächengewichte bestehen.
Die bisherigen Präzisionsfälle werden bei absoluten Fehlern mit der
tatsächlichen Unsicherheit σ/√w aufgerufen; dies erhält dieselbe Fit-Metrik.

**Abgrenzung des relativen Modus:** Bei absolute_uncertainty=False oder ohne
vorgegebene Unsicherheiten bleibt das bisherige Modell
Var(I_p) = γ σ_p²/w_p mit γ = Q/(N−K) bestehen. Es interpretiert die
kombinierten Gewichte als relative Präzision. Seine Kovarianz und F-Tests
sind nur unter dieser Annahme kalibriert. Für geometrische Gewichte mit
unbekanntem Messrauschen wurde keine neue Skalen- oder Referenzverteilung
eingeführt. Vier strict-xfail-Prüffälle halten diese nicht unterstützte
Interpretation weiterhin sichtbar; die fünf bisherigen xfail-Fälle mit
absoluten Messunsicherheiten bestehen jetzt regulär.

**Validierung nach der Kovarianzkorrektur, vor der Holm-Ergänzung:**
Die damalige Testsuite lieferte
**123 bestandene Tests und vier erwartete Fehler**. Die gezielten
statistischen Prüffälle und die zugehörigen Kovarianzgleichungen bestehen
auch bei Behandlung von Warnungen als Fehler. Der erweiterte
Standard-Analyzer-Test bestätigt die Fortpflanzung bis zu den T_n-Statistiken
und radialen Fehlerbändern bei interner Pixelintegration. Im Gesamt-Testlauf
wurden weiterhin Laufzeit- und Plotwarnungen ausgegeben; alle numerischen
Vergleichsprüfungen bestanden. Die Warnungen sind nicht unterdrückt.
Das Supplement beschreibt beide Kovarianzmodelle und wurde neu kompiliert.

#### Vergleich an den experimentellen Daten (Codex, 10. November 2026)

Alle sechs Standardanalysen wurden mit ihren bisherigen q-Bereichen,
Basisgrößen (25, 25), Gewichtungen und Zentrierungseinstellungen erneut
ausgeführt. Die Ergebnisgrafiken sind aktualisiert. Der Vergleich der
Kovarianzen erfolgt am jeweils selben Fit; damit werden ausschließlich
Fehlerfortpflanzung und Blockstatistiken verglichen.

Bei den vier Nanoflower-Datensätzen sinken die Leistungsfehlerbalken im
Median um 0.92–1.19 %, während die T_n-Werte im Median um 1.42–2.35 %
steigen. Die radialen Fehlerbänder werden über das ganze q-Gitter im Median
nur um etwa 0.02 % schmaler; nahe den Annulusgrenzen treten lokal
Reduktionen bis 16.79 % auf. Die QR-Kontrollrechnung bestätigt, dass die
Unterschiede von den geometrischen Gewichten kommen; die numerische
Faktorisierungsänderung ist in diesen Daten vernachlässigbar.

Nur bei 10 mT ändert sich eine nominelle 95-%-Entscheidung: cos 5 wechselt
von p = 0.07070 zu p = 0.04839. Nach Holm-Korrektur über alle 51 Blocktests
ändert sich dagegen in keinem der sechs Datensätze die Menge signifikanter
Blöcke. Insbesondere liefert der cos-5-Wechsel keinen zusätzlichen
Holm-signifikanten Strukturbeitrag.

Die zwei PRL-Benchmarks verwenden keine Flächengewichte und bleiben bis
auf Rundungsfehler unverändert. Es treten bei keinem der sechs Vergleiche
numerische Rangwechsel auf. Details, quantitative Tabellen,
Reproduktionsbefehle und Vergleichsgrafiken stehen im
[experimentellen Vergleichsbericht](../experimental_data/README.md#covariance-correction-comparison).
Die Holm-Auswertung ist hier eine zusätzliche Kontrolle der vorhandenen
Block-p-Werte; eine Strukturauswahl mit reduziertem Refit wurde nicht
eingeführt.

---

## 3. Ungerade Ordnungen: Zentrierungsdiagnose im Koeffizientenraum

### Warum

Die ungeraden Koeffizienten dienen als Qualitätsmaß, sofern keine chiralen
Beiträge erlaubt sind. Bisher wird dafür eine Fraktion berechnet, die
rauschverzerrt ist (Punkt 2), und ein gemeinsamer Test über alle ungeraden
Blöcke ist unempfindlich:

- Gemeinsamer Wald-Test über alle ungeraden Moden bei 10 mT: p = 0.11
  (676 Freiheitsgrade).
- Blocktest für sin 1 allein: p ≈ 0.

Ein echtes Signal geht unter vielen Freiheitsgraden unter. Es braucht deshalb
einen Test mit **physikalisch motivierter Richtung**.

Eine Verschiebung des Zentrums um δ = (δ_z, δ_y) ergibt in erster Ordnung

I_mess(**q**) ≈ I(**q**) − δ · ∇I(**q**).

- Für den isotropen Anteil gilt δ·∇a₀ = a₀′(q)(δ_z cos θ + δ_y sin θ). Die
  Verschiebung erzeugt also cos 1 und sin 1 mit dem **radialen Profil
  −a₀′(q)**: a₁ᶜ = −δ_z a₀′ und a₁ˢ = −δ_y a₀′.
- Jede gerade Ordnung n koppelt an n ± 1. Bei anisotropen Mustern entstehen
  daher auch n = 3, 5, … mit vorhersagbarer Form.
- Gerade Ordnungen ändern sich erst in zweiter Ordnung in δ.

Eine Verschiebung sagt also mit nur **zwei Parametern** das gesamte ungerade
Muster einschließlich seiner radialen Profile voraus. Das macht sie
falsifizierbar.

Plausibilitätscheck bei 10 mT mit Basis (4,16): Das gemessene a₁ˢ(q) korreliert
mit −a₀′(q) (r = 0.83). Die Anpassung ergibt δ_y ≈ 2.8·10⁻⁴ nm⁻¹, etwa
0.1 Pixel. Die Vorzeichenkonvention ist noch zu prüfen. Das punktweise
normierte RMS sinkt von 2.4 auf 1.1. Eine Verschiebung um etwa 0.1 Pixel
entlang q_y erklärt also den Großteil des sin-1-Signals. Falls q_y bei diesem
Aufbau vertikal ist, kommt auch die Schwerkraft als Ursache in Betracht.

### Wie

Neue Funktion, zum Beispiel `odd_mode_shift_test(coefficients, covariance,
indices, q_min, q_max)` in `structure.py`:

1. **Gerader Teil:** Î_even = Σ_{|n| gerade} ĉ_nm Φ_nm.
2. **Analytischer Gradient:** ∂_q R_m über die Legendre-Ableitung
   (`numpy.polynomial.legendre.legder`, Kettenregel dx/dq = 4q/Δ), ∂_θ T_n
   trivial. Daraus ∂I/∂q_z und ∂I/∂q_y in Polarkoordinaten. Finite Differenzen
   vermeiden, weil `radial_mode` Werte außerhalb von [q_min, q_max] ablehnt.
3. **Projektion auf die ungeraden Moden** per Quadratur, Gauß-Legendre in x und
   gleichmäßig in θ. Dank Orthonormalität ist das exakt:
   g_z,j = ⟨−∂_z Î_even, Φ_j⟩ und g_y,j = ⟨−∂_y Î_even, Φ_j⟩ für alle
   ungeraden j. So entsteht eine Matrix G (k_odd × 2).
4. **GLS für δ** mit der ungeraden Kovarianz C_odd:
   δ̂ = (Gᵀ C⁺ G)⁻¹ Gᵀ C⁺ ĉ_odd, Cov(δ̂) = (Gᵀ C⁺ G)⁻¹. Die Pseudoinverse wie in
   `block_test._block` mit derselben SVD-Schwelle bilden.
5. **Rest-Test:** r = ĉ_odd − Gδ̂, T = rᵀ C⁺ r mit (Rang − 2) Freiheitsgraden,
   bei geschätzter Skala als F-Test.
6. **Rückgabe:** δ̂ ± σ_δ (in q-Einheiten und, falls die Geometrie bekannt ist,
   in Pixeln), T_shift für die Hypothese δ = 0, T_rest und die Profile a₁(q)
   gegen −δ̂·a₀′(q) für einen Plot.

Optional können weitere Störsignaturen als zusätzliche Spalten in G
aufgenommen werden, etwa ein linearer Untergrund- oder Effizienzgradient über
den Detektor. Dieser erzeugt ebenfalls n = 1, aber mit anderem radialem Profil
(nicht ∝ a₀′). Über die Profilform lassen sich die Ursachen trennen.

### Vorteil

- Die ungerade Struktur wird in Ursachen zerlegt: „durch eine Verschiebung
  erklärt“ (mit δ̂ ± σ) oder „ungerader Rest“, also chirale Beiträge, Probleme
  bei der Datenreduktion oder Detektoreffekte.
- Die Zentrierung bekommt eine **Unsicherheit**. Die bisherige Paritätszentrierung
  im Datenraum liefert keine, und ihre Interpolationskorrelationen werden nicht
  propagiert.
- Es wird nichts interpoliert. Die Diagnose läuft vollständig auf dem bestehenden
  Fit und kann die Datenraum-Zentrierung prüfen oder vorbereiten.
- Bei Kanälen mit chiralen Beiträgen ist der Rest-Test die richtige Größe: Eine
  Verschiebung allein kann ein chirales Muster im Allgemeinen nicht nachbilden.

### Einschränkungen

- Erste Ordnung in δ. Bei Verschiebungen im Bereich von Subpixeln ist das
  unkritisch.
- Die Unsicherheit von G aus den geraden Koeffizienten wird vernachlässigt.
  Sie ist zweiter Ordnung, solange die geraden Moden gut bestimmt sind.
- Der Gradient der trunkierten Expansion ist nahe q_min und q_max empfindlich.
  Gegebenenfalls nur einen inneren q-Bereich verwenden oder m begrenzen.

### Tests

- Synthetisches isotropes Muster, um bekanntes δ verschoben: δ̂ trifft δ innerhalb
  der Fehler, und T_rest ist unauffällig.
- Anisotropes Muster (cos 0 + cos 2), verschoben: Die vorhergesagten Anteile in
  n = 1 und n = 3 stimmen.
- Muster mit eingebautem chiralem bzw. ungeradem Term ohne Verschiebung:
  T_rest ist signifikant.

### Einschätzung von Codex

**Die wissenschaftlich interessanteste Erweiterung, aber aufwendiger als hier
angesetzt.** Die Taylor-Entwicklung und die Kopplung n → n ± 1 sind richtig.
Eine physikalisch vorgegebene Verschiebungssignatur kann wesentlich
empfindlicher sein als ein Test über sämtliche ungeraden Richtungen. Vor einer
Umsetzung sind folgende Punkte zu lösen:

1. **Die tatsächliche Fit-Abbildung verwenden.** Die kontinuierliche Projektion
   des Gradienten auf die orthonormale Basis ist bei diskreter, maskierter
   WLS-Abtastung im Allgemeinen nicht die Antwort der gefitteten
   Koeffizienten auf eine Verschiebung. Bei festem Design X und Gewicht W
   lautet diese Antwort

   $$
   G_{\mathrm{fit}}
   = S_{\mathrm{odd}}(X^{T}WX)^{-1}X^{T}W D,
   \qquad
   D = \left(-\partial_z\widehat I_{\mathrm{even}},
             -\partial_y\widehat I_{\mathrm{even}}\right).
   $$

   S_odd wählt die ungeraden Koeffizienten aus. D muss an denselben Pixeln
   bzw. mit derselben Pixelintegration wie X berechnet werden; numerisch
   sollte derselbe QR-/SVD-Löser wie beim Fit verwendet werden. Die Formel
   gilt für die Antwort bei festem Design und festem W. In einer kleinen
   synthetischen Gegenprüfung wich die kontinuierliche Projektion um rund
   23 % von dieser diskreten Fit-Antwort ab. Auch die Faktoren zwischen
   orthonormalen Winkelmoden und den Fourier-Amplituden a_n müssen
   berücksichtigt werden.

2. **Quadratur auf Konvergenz prüfen.** Orthonormalität macht die
   Gradientenprojektion nicht automatisch exakt. Durch q und 1/q in den
   Ableitungen entstehen nach der Transformation x ∝ q² im Allgemeinen
   nichtpolynomiale Integranden; eine endliche Gauß-Quadratur muss deshalb
   separat konvergiert werden.

3. **Den identifizierbaren Rang verwenden.** Die zwei Verschiebungsrichtungen
   können Rang 0, 1 oder 2 haben. Der Rest-Test hat bei festem G und passendem
   Gaußschen Kovarianzmodell die Freiheitsgrade

   $$
   \nu_{\mathrm{rest}}
   = \operatorname{rank}(C_{\mathrm{odd}})
     - \operatorname{rank}\!\left[(C_{\mathrm{odd}}^{+})^{1/2}G\right],
   $$

   nicht pauschal Rang − 2. Rang und Pseudoinversen müssen konsistente
   Toleranzen verwenden. Nicht identifizierbare Verschiebungsrichtungen
   lassen sich auch durch eine Pseudoinverse nicht physikalisch bestimmen.

4. **Die Unsicherheit von G berücksichtigen.** G wird aus geschätzten geraden
   Koeffizienten aufgebaut. Gerade und ungerade Koeffizienten können durch
   Maskierung und Gewichtung korreliert sein. Die angegebenen Kovarianz- und
   Testformeln gelten zunächst bedingt auf ein festes G; dessen Unsicherheit
   ist nicht allgemein nur zweiter Ordnung. Erforderlich sind eine gemeinsame
   Unsicherheitsrechnung oder passend kalibrierte Simulationen des gesamten
   Verfahrens.

5. **Linearitätsbereich und Interpretation prüfen.** Eine Verschiebung um
   einen Bruchteil eines Pixels garantiert noch keine gültige Linearisierung;
   maßgeblich sind die lokale Variationslänge und die Krümmung des Signals.
   Ein physikalischer ungerader Beitrag im Verschiebungsunterraum ist von
   einer Verschiebung nicht unterscheidbar. Ein signifikanter Rest verwirft
   „Verschiebung allein“, identifiziert aber keine Ursache; ein unauffälliger
   Rest schließt Chiralität nicht aus. Der dritte synthetische Test muss
   deshalb einen ungeraden Beitrag außerhalb des Verschiebungsunterraums
   verwenden.

Die Kanalinterpretation muss vor jeder Zentrierung feststehen: Laut
[Datensatzbeschreibung](../experimental_data/doi_10.48550_arXiv.2606.18791/README.md)
enthalten 10 und 30 mT den Mittelwert der beiden Spin-Flip-Kanäle, 100 mT und
1 T dagegen nur den +−-Kanal. Bei Kanälen mit erlaubten chiralen Beiträgen
sollte diese Diagnose keine automatische Entfernung ungerader Struktur
auslösen. Eine Zuordnung zur Schwerkraft benötigt zusätzlich die
Instrumentgeometrie und unabhängige Informationen zum Strahlzentrum.

---

## 4. Trunkierung: Informationskriterien als Ergänzung

### Warum

Durch die Auswahl über Signifikanz (Punkt 1) und die Rauschkorrektur (Punkt 2)
ist eine große Basis unkritisch. Die Wahl von (n_max, m_max) bleibt aber für zwei
Dinge relevant: die **Rechenzeit** und die **Stabilität der Schlussfolgerungen**.
Bei 10 mT braucht das Signal radial viel, im Winkel wenig:

| (n_max, m_max) | K | χ² | AIC | BIC |
|---|---|---|---|---|
| (4, 8)   | 81   | 13038 | 13200 | 13766 |
| (4, 16)  | 153  | 7296  | **7602** | **8671** |
| (12, 12) | 325  | 8017  | 8667  | 10938 |
| (25, 25) | 1326 | 6087  | 8739  | 18004 |

`mode_convergence_standard_analyzer` vergleicht nur mit dem größten getesteten
Modell und nutzt ungewichtete Normen. Es kann daher nicht feststellen, dass
(4, 16) die Daten effizienter beschreibt.

### Wie

- AIC = χ² + 2K und BIC = χ² + K ln N auf dem Gitter, das die
  Konvergenzfunktion ohnehin berechnet. Dafür das **gewichtete** χ² verwenden.
- Optional: einen F-Test für verschachtelte Rechtecke als zusätzliches Kriterium.
- Eine **Stabilitätsprüfung der Schlussfolgerungen**: Man prüft, ob die Auswahl
  aus Punkt 1, also die Menge signifikanter Blöcke, über mehrere
  (n_max, m_max) und gegebenenfalls über leicht verschobene q-Bereiche gleich
  bleibt. Das ist die eigentliche Antwort auf „stabil enthalten“.

### Vorteil

Die Basisgröße wird datengetrieben begründet, und die Robustheit der
Kernaussagen lässt sich direkt zeigen, statt sie nur numerisch zu konvergieren.

### Einschätzung von Codex

**Als Ergänzung sinnvoll, sobald Likelihood und Skalenbehandlung feststehen.**
Die Stabilität der Schlussfolgerungen über mehrere Basisgrößen zu prüfen ist
besonders nützlich. Signifikanzkorrektur und Rauschabzug machen eine beliebig
große Basis jedoch nicht unkritisch: Konditionierung, Schätzervarianz und
Verlässlichkeit der Kovarianz bleiben relevant.

Die Formeln AIC = Q + 2K und BIC = Q + K ln N gelten bis auf gemeinsame
Konstanten für eine Gaußsche Likelihood mit bekannter, fester
Fehlerkovarianz. Ein durch zusätzliche geometrische Gewichte definiertes
WLS-Q ist nicht automatisch ein solches Likelihood-Kriterium. Wird eine
gemeinsame unbekannte Varianzskala aus den Residuen geschätzt, lautet der
Fitanteil stattdessen N ln(Q/N); die Skala zählt als zusätzlicher Parameter:

$$
\mathrm{AIC}=N\ln(Q/N)+2(K+1)+\mathrm{const},
\qquad
\mathrm{BIC}=N\ln(Q/N)+(K+1)\ln N+\mathrm{const}.
$$

Dabei ist K die Anzahl identifizierbarer Modellparameter. Die Unterscheidung
zwischen bekannter und geschätzter Skala erläutert auch die
[R-Dokumentation zu extractAIC](https://stat.ethz.ch/R-manual/R-devel/library/stats/html/extractAIC.html).

Informationskriterien sollten auf denselben Beobachtungen und mit denselben
Rauschannahmen verglichen werden. Leicht veränderte q-Bereiche sind ein guter
Robustheitstest, ihre AIC-/BIC-Werte aber kein unmittelbarer Modellvergleich,
weil sich dabei die Datenmenge ändert. Ein F-Test nach datengesteuerter
Auswahl eines Rechtecks benötigt wiederum die unter Punkt 1 beschriebene
Berücksichtigung der Auswahl.

---

## 5. Basisunabhängigkeit belegen: Referenz mit Fourier pro Ring

### Warum

Die radiale Legendre-Basis ist eine Glattheitsannahme, keine kanonische Wahl.
Innerhalb desselben Polynomraums sind Rekonstruktion und P_n basisunabhängig,
denn Σ_m c² ist invariant unter orthogonalen Drehungen. Zwischen
unterschiedlichen Unterräumen ist keine Rekonstruktion „wahrer“. Prüfen lässt
sich aber, ob die Glättung Struktur **erfindet**.

### Wie

Eine Funktion `ring_fourier_reference(data, q_edges, n_max)`: Für jeden q-Ring
die Winkelkoeffizienten a_n, b_n per WLS mit Fehlern bestimmen, ohne radiale
Annahme. Im Spektrenplot optional als Punkte mit Fehlerbalken über I_n(q)
legen, mit der Normierung aus `eq:fourier-amplitudes` im Supplement.

### Vorteil

Ein günstiger, sehr überzeugender Validierungsplot: Liegen die geglätteten
Spektren innerhalb der Fehler auf den Ringwerten, ist die Darstellung von der
Basiswahl unabhängig belegt. Das ist auch ein gutes Argument für Gutachter.

### Einschätzung von Codex

**Ein guter, früh umsetzbarer Robustheitstest.** Ringweise Fourier-Fits
vermeiden die globale radiale Legendre-Trunkierung und können deren Einfluss
sichtbar machen. Sie sind dennoch nicht frei von radialen Annahmen:
Ringbreite, ein innerhalb des Rings konstantes Profil sowie Winkeltrunkierung,
Maske und Gewichtung bestimmen ihre Auflösung und mögliche Verzerrungen.

Für den Vergleich sollte die globale Rekonstruktion durch denselben
Ring-Schätzer geführt oder mit den entsprechenden Bin-Mitteln verglichen
werden. Das globale Spektrum lediglich in der Ringmitte auszuwerten kann bei
starken radialen Gradienten oder ungleichmäßiger Winkelabdeckung einen
scheinbaren Unterschied erzeugen. Die Fourier-Normierung muss dabei mit
der Gleichung **eq:fourier-amplitudes** im Supplement übereinstimmen.

Beide Schätzungen beruhen auf denselben Daten und haben korrelierte Fehler.
Überlappende Einzel-Fehlerbalken sind deshalb kein unabhängiger
Validierungstest; ein quantitativer Differenztest benötigt ihre
Kreuzkovarianz. Gute Übereinstimmung belegt Robustheit gegenüber den
geprüften radialen Darstellungen und Auflösungen, keine vollständige
Basisunabhängigkeit.

---

## 6. Kleinere Punkte

- **Achsenbeschriftung** im Leistungspanel: `P_n^2/P_max^2` in `P_n/P_max`
  ändern. Dann kann die Erklärung im Supplement entfallen.
- **Plot optional machen:** `standard_analyzer` bleibt die Haupt-Funktion für
  Nutzer und erzeugt standardmäßig den Plot. Ein Parameter
  `output_path=None` → kein Plot würde aber Sweeps und Tests deutlich schneller
  machen. Ohne große Umstrukturierung ließe sich der Plotteil in eine private
  Funktion `_plot_standard_analysis(result, ...)` auslagern.
- **Mehrfachtests im Plot:** neben der 95-%-Linie die Holm-Schwelle aus Punkt 1
  anzeigen.
- **Supplement nachziehen:** Neue Größen (rauschkorrigierte Leistungen und
  Verschiebungstest) nach dem bisherigen Muster als
  Gleichungen mit zugehörigen Tests in `test_supplementary_equations.py`
  dokumentieren.

### Einschätzung von Codex

**Die praktischen Änderungen sind sinnvoll und können früh erfolgen.**
Eine verständliche Leistungsachse und optionales Rendern vereinfachen Nutzung,
Tests und Parametersweeps. Bei einem Aufruf ohne Plot müssen sämtliche
numerischen Diagnosen weiterhin berechnet werden; der Rückgabewert für
**output_path** muss einen fehlenden Pfad erlauben.

Eine Holm-Korrektur lässt sich im Allgemeinen nicht durch eine einzige
zusätzliche horizontale T-Schwelle darstellen: Sie ist schrittweise, und
Blöcke können unterschiedliche numerische Ränge haben. Verständlicher sind
korrigierte p-Werte oder die jeweils zutreffenden Einzelentscheidungen.

Neue Gleichungen und synthetische Regressionstests sollten gemeinsam mit den
validierten Implementierungen ins Supplement übernommen werden. Die unter
den Punkten 1–5 genannten Voraussetzungen gehören dabei ausdrücklich zur
Definition und Interpretation der neuen Kennzahlen.

## Zusammenfassung

| # | Änderung | Beantwortet | Aufwand |
|---|---|---|---|
| 1 | Auswahl über Signifikanz und reduzierter Refit | Was ist stabil enthalten? | mittel |
| 2 | Rauschkorrektur der Leistungen und Fraktionen | Was liegt über dem Rauschen? | gering |
| 3 | Verschiebungstest der ungeraden Moden | Ist die ungerade Struktur Zentrierung oder etwas anderes? | mittel |
| 4 | AIC/BIC und Stabilität der Auswahl | Ist die Basisgröße angemessen, sind die Aussagen robust? | gering |
| 5 | Fourier-Referenz pro Ring | Erfindet die radiale Glättung Struktur? | gering |

### Priorisierung von Codex

Die Vorschläge erweitern die Analyse sinnvoll, wenn **Signalstärke,
statistische Signifikanz und Stabilität** als unterschiedliche Aussagen
erhalten bleiben. Die bisherige Parseval-Auswertung sollte als beschreibende
Grundlage weiter verfügbar sein.

1. **Reproduzierbarkeit und Kovarianz klären:** Ad-hoc-Skripte samt
   Konfiguration aufnehmen, Arbeitskovarianz und tatsächliches Rauschmodell
   abgleichen sowie numerische Konditionierung prüfen. Optionale Plots und
   Achsenbeschriftung können parallel dazu als kleine Änderungen erfolgen.
2. **Rauschdiagnostik und Holm-Korrektur ergänzen:** Rohleistungen erhalten,
   zusätzliche Größen eindeutig benennen und Fehlerraten synthetisch prüfen.
3. **Ringvergleich und Informationskriterien ergänzen:** AIC/BIC können
   hinzukommen, sobald Likelihood und Skalenbehandlung festgelegt sind.
4. **Verschiebungstest entwickeln und kalibrieren:** Diskrete Fit-Antwort,
   Rang, Unsicherheit von G und Kanalphysik gemeinsam berücksichtigen. Dies
   ist der anspruchsvollste neue methodische Baustein.

