import * as tf from '@tensorflow/tfjs';

const predictButton = document.getElementById('predictButton');
const resultElement = document.getElementById('result');

let model;

async function loadModel() {
    // Simulate loading a model. In a real scenario, you'd load a pre-trained model.
    // For this demo, we'll create a simple linear model.
    model = tf.sequential();
    model.add(tf.layers.dense({units: 1, inputShape: [1]}));
    model.compile({loss: 'meanSquaredError', optimizer: 'sgd'});

    // Train a simple model for demonstration
    const xs = tf.tensor2d([1, 2, 3, 4], [4, 1]);
    const ys = tf.tensor2d([1, 3, 5, 7], [4, 1]);
    await model.fit(xs, ys, {epochs: 100});

    console.log('Dummy model loaded and trained.');
}

async function runPrediction() {
    if (!model) {
        resultElement.innerText = 'Result: Model not loaded yet. Please wait.';
        return;
    }

    // Simulate an input for prediction
    const input = tf.tensor2d([5], [1, 1]);
    const prediction = model.predict(input);
    const predictionValue = (await prediction.data())[0];

    resultElement.innerText = `Result: Predicted value for input 5 is approximately ${predictionValue.toFixed(2)}`;
    console.log('Prediction made:', predictionValue);
}

// Load model when the page loads
window.onload = loadModel;

predictButton.addEventListener('click', runPrediction);
