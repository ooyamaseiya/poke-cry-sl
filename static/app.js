document.addEventListener('DOMContentLoaded', () => {
    const fileInput = document.getElementById('audioFile');
    const predictBtn = document.getElementById('predictBtn');
    const loadingEl = document.getElementById('loading');
    const resultArea = document.getElementById('resultArea');
    const predictionList = document.getElementById('predictionList');
    const errorArea = document.getElementById('errorArea');

    function showError(message) {
        errorArea.textContent = message;
        errorArea.classList.remove('hidden');
        loadingEl.classList.add('hidden');
        resultArea.classList.add('hidden');
    }

    function hideError() {
        errorArea.classList.add('hidden');
    }

    predictBtn.addEventListener('click', async () => {
        const files = fileInput.files;
        if (files.length === 0) {
            alert("音声ファイルを選択してください。");
            return;
        }

        // Reset UI state
        hideError();
        resultArea.classList.add('hidden');
        predictionList.innerHTML = '';
        loadingEl.classList.remove('hidden');
        predictBtn.disabled = true;

        const formData = new FormData();
        formData.append('file', files[0]);

        try {
            const response = await fetch('/predict', {
                method: 'POST',
                body: formData,
            });

            if (!response.ok) {
                const errorData = await response.json().catch(() => ({ detail: "通信エラーが発生しました。" }));
                throw new Error(errorData.detail || `HTTP Error ${response.status}`);
            }

            const data = await response.json();

            if (data.success && data.predictions) {
                // Show results
                loadingEl.classList.add('hidden');
                resultArea.classList.remove('hidden');
                
                data.predictions.forEach(pred => {
                    const li = document.createElement('li');
                    li.className = 'prediction-item';
                    li.innerHTML = `
                        <div class="info-row">
                            <span>#${pred.rank} ${pred.name}</span>
                            <span>${pred.probability}%</span>
                        </div>
                        <div class="progress-bar-container">
                            <div class="progress-bar" style="width: ${pred.probability}%"></div>
                        </div>
                    `;
                    predictionList.appendChild(li);
                });
            } else {
                throw new Error("推論に失敗しました。");
            }

        } catch (error) {
            console.error(error);
            showError(error.message);
        } finally {
            predictBtn.disabled = false;
        }
    });
});
