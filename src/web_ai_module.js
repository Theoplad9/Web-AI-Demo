// Web-AI-Demo - Advanced TensorFlow.js Module

import * as tf from '@tensorflow/tfjs';

tf.setBackend('webgl');

class WebAIModel {
    constructor() {
        this.model = null;
        this.isModelLoaded = false;
        console.log('WebAIModel initialized.');
    }

    async loadPretrainedModel(modelUrl = 'https://storage.googleapis.com/tfjs-models/tfjs/mobilenet_v2_1.0_224/model.json') {
        console.log(`Attempting to load model from: ${modelUrl}`);
        try {
            this.model = await tf.loadGraphModel(modelUrl);
            this.isModelLoaded = true;
            console.log('Pre-trained model loaded successfully!');
        } catch (error) {
            console.error('Failed to load pre-trained model:', error);
            this.model = this.createSimpleDummyModel();
            this.isModelLoaded = true;
            console.warn('Using a simple dummy model as fallback.');
        }
    }

    createSimpleDummyModel() {
        const dummyModel = tf.sequential();
        dummyModel.add(tf.layers.dense({ units: 10, activation: 'relu', inputShape: [784] }));
        dummyModel.add(tf.layers.dense({ units: 10, activation: 'softmax' }));
        dummyModel.compile({ optimizer: 'adam', loss: 'categoricalCrossentropy', metrics: ['accuracy'] });
        console.log('Simple dummy model created.');
        return dummyModel;
    }

    async preprocessImage(imageData) {
        console.log('Preprocessing image data...');
        const imageTensor = tf.browser.fromPixels(imageData)
            .resizeNearestNeighbor([224, 224])
            .toFloat()
            .div(tf.scalar(255.0))
            .expandDims();
        console.log('Image preprocessed. Tensor shape:', imageTensor.shape);
        return imageTensor;
    }

    async predict(inputData) {
        if (!this.isModelLoaded || !this.model) {
            console.error('Model is not loaded. Cannot make predictions.');
            return null;
        }
        console.log('Making prediction...');
        const prediction = this.model.predict(inputData);
        const result = await prediction.data();
        prediction.dispose();
        console.log('Prediction complete. Result:', result);
        return result;
    }

    async runExamplePrediction() {
        console.log('Running example prediction...');
        const dummyInput = tf.randomNormal([1, 224, 224, 3]);
        const predictions = await this.predict(dummyInput);
        if (predictions) {
            const top5 = Array.from(predictions)
                .map((p, i) => ({ probability: p, className: `class_${i}` }))
                .sort((a, b) => b.probability - a.probability)
                .slice(0, 5);
            console.log('Top 5 predictions:', top5);
            return top5;
        }
        return null;
    }
}

const webAI = new WebAIModel();

document.addEventListener('DOMContentLoaded', async () => {
    await webAI.loadPretrainedModel();

    const predictButton = document.getElementById('predictButton');
    const resultElement = document.getElementById('result');

    if (predictButton && resultElement) {
        predictButton.addEventListener('click', async () => {
            resultElement.innerText = 'Predicting...';
            const predictions = await webAI.runExamplePrediction();
            if (predictions) {
                resultElement.innerHTML = 'Result: <br>' + predictions.map(p => `Class ${p.className}: ${p.probability.toFixed(4)}`).join('<br>');
            } else {
                resultElement.innerText = 'Result: Prediction failed.';
            }
        });
    }
});

export default WebAIModel;
