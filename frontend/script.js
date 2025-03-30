async function analyzeSentiment() {
    const tweet = document.getElementById("tweetInput").value;
    if (!tweet.trim()) return alert("Bitte gib einen Tweet-Text ein.");

    try {
        const response = await fetch("/predict", {
            method: "POST",
            headers: {
                "Content-Type": "application/json"
            },
            body: JSON.stringify({ text: tweet })
        });

        const result = await response.json();

        document.getElementById("biasLabel").textContent = labelToText(result.label);
        document.getElementById("confidence").textContent = (result.confidence * 100).toFixed(2) + "%";
        document.getElementById("resultBox").classList.remove("d-none");
    } catch (error) {
        console.error("Fehler bei der Analyse:", error);
        alert("Analyse fehlgeschlagen. Bitte versuche es erneut.");
    }
}

function labelToText(label) {
    switch (label) {
        case 0:
            return "Pro Russland 🇷🇺";
        case 1:
            return "Neutral ⚖️";
        case 2:
            return "Pro Ukraine 🇺🇦";
        default:
            return "Unbekannt";
    }
}

// Abrufen des aktuellen Bias-Scores beim Laden
document.addEventListener("DOMContentLoaded", async () => {
    try {
        const response = await fetch("/bias_score");
        const data = await response.json();

        const biasScore = data.bias_score;
        console.log("Bias Score:", biasScore); // ✅ hier ist er gültig

        const indicator = document.getElementById("biasIndicator");
        const percent = ((biasScore + 1) / 2) * 100;

        indicator.style.left = `${percent}%`;
        indicator.style.transform = "translateX(-50%)";

        document.getElementById("biasScoreLabel").textContent = `Bias-Score: ${biasScore.toFixed(2)}`;
    } catch (error) {
        console.warn("Bias-Score konnte nicht geladen werden.", error);
    }
    fetch("/training_metadata")
        .then(res => res.json())
        .then(meta => {
            document.getElementById("metaModel").textContent = meta.model_name;
            document.getElementById("metaDate").textContent = meta.trained_on;
            document.getElementById("metaOriginal").textContent = meta.original_tweets.toLocaleString() + " Tweets";
            document.getElementById("metaAugmented").textContent = meta.augmented_tweets.toLocaleString() + " Tweets";
            document.getElementById("metaAcc").textContent = (meta.val_accuracy * 100).toFixed(1) + " %";
            document.getElementById("metaSize").textContent = meta.model_size_mb + " MB";
        })
        .catch(err => console.warn("Fehler beim Laden der Metadaten:", err));
    // Verteilung der Trainingsdaten laden & anzeigen
    fetch("/label_distribution")
        .then(res => res.json())
        .then(data => {
            const counts = [0, 0, 0];
            data.forEach(item => {
                counts[item._id] = item.count;
            });

            const ctx = document.getElementById("labelChart").getContext("2d");
            new Chart(ctx, {
                type: "bar",
                data: {
                    labels: ["🇷🇺 Pro Russland", "⚖️ Neutral", "🇺🇦 Pro Ukraine"],
                    datasets: [{
                        label: "Anzahl Trainings-Tweets",
                        data: counts,
                        backgroundColor: ["#d9534f", "#f0ad4e", "#5bc0de"]
                    }]
                },
                options: {
                    responsive: true,
                    plugins: {
                        legend: { display: false },
                        tooltip: {
                            callbacks: {
                                label: ctx => `${ctx.formattedValue} Tweets`
                            }
                        }
                    },
                    scales: {
                        y: {
                            beginAtZero: true,
                            ticks: {
                                precision: 0,
                                stepSize: 50
                            }
                        }
                    }
                }
            });
        })
        .catch(err => console.warn("Fehler beim Laden der Label-Verteilung:", err));

    // Zufallsbeispiel laden
    document.getElementById("loadExampleBtn").addEventListener("click", async () => {
        try {
            const res = await fetch("/random_training_example");
            const data = await res.json();

            document.getElementById("exampleText").textContent = data.text;
            document.getElementById("exampleLabel").textContent = data.label;
            document.getElementById("exampleBox").classList.remove("d-none");
        } catch (err) {
            console.warn("Fehler beim Laden eines Beispiels:", err);
            alert("Beispiel konnte nicht geladen werden.");
        }
        console.log("Button wurde geklickt");
    });
});
