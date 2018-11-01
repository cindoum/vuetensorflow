<template>
  <div>
    <div class="train-controls">
      <button class="button-add-example button--green" v-on:click="start()">Predict mobilenet</button>
      <img id="initial_pic" src="~assets/plane.jpg" width="224" height="224" />
      <pre>{{result}}</pre>
    </div>
  </div>
</template>

<script>
import * as tf from "@tensorflow/tfjs";
import {IMAGENET_CLASSES} from './kls.js';


export default {
  data() {
    return {
      mobilenet: null,
      result: null
    };
  },
  methods: {
    async start() {
      this.mobilenet = await tf.loadModel(
        "https://storage.googleapis.com/tfjs-models/tfjs/mobilenet_v1_0.25_224/model.json"
      );

      this.mobilenet.predict(tf.zeros([1, 224, 224, 3])).dispose();

      const pic = document.getElementById('initial_pic');

      if (pic) {
        this.predict(pic)
      }
    },
    async predict(imgElement) {
      const logits = tf.tidy(() => {
          // tf.fromPixels() returns a Tensor from an image element.
          const img = tf.fromPixels(imgElement).toFloat();

          const offset = tf.scalar(127.5);
          // Normalize the image from [0, 255] to [-1, 1].
          const normalized = img.sub(offset).div(offset);

          // Reshape to a single-element batch so we can pass it to predict.
          const batched = normalized.reshape([1, 224, 224, 3]);

          // Make a prediction through mobilenet.
          return this.mobilenet.predict(batched);
        });

       const classes = await this.getTopKClasses(logits, 10);
       this.result = classes;
      },
      async getTopKClasses(logits, topK) {
        const values = await logits.data();

        const valuesAndIndices = [];
        for (let i = 0; i < values.length; i++) {
          valuesAndIndices.push({value: values[i], index: i});
        }
        valuesAndIndices.sort((a, b) => {
          return b.value - a.value;
        });
        const topkValues = new Float32Array(topK);
        const topkIndices = new Int32Array(topK);
        for (let i = 0; i < topK; i++) {
          topkValues[i] = valuesAndIndices[i].value;
          topkIndices[i] = valuesAndIndices[i].index;
        }

        const topClassesAndProbs = [];
        for (let i = 0; i < topkIndices.length; i++) {
          topClassesAndProbs.push({
            className: IMAGENET_CLASSES[topkIndices[i]],
            probability: topkValues[i]
          })
        }
        return topClassesAndProbs;
      }

    }
  }
</script>

<style>
.field,
.field-label {
  height: 30px;
  float: left;
  padding: 0px 15px;
  float: left;
  width: 50%;
}

.field {
  border-radius: 0px 5px 5px 0px;
  border: 1px solid #eee;
  margin-bottom: 15px;
  height: 40px;
}

.col-sm-1:after {
  content: "";
  display: table;
  clear: both;
}

.section,
.field-label {
  text-align: left;
  font-family: "Quicksand", "Source Sans Pro", -apple-system, BlinkMacSystemFont,
    "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif; /* 1 */
  font-weight: 100;
}

.field-label {
  font-weight: 700;
}

.button-add-example {
  width: 100%;
  margin-bottom: 10px;
  cursor: pointer;
}

.button-train {
  width: 100%;
}

.predict-controls {
  padding-top: 30px;
  padding-bottom: 30px;
}

.predict-controls .element {
  width: 50%;
  display: block;
}

button {
  margin-top: 10px;
  font-family: "Quicksand", "Source Sans Pro", -apple-system, BlinkMacSystemFont,
    "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif; /* 1 */
  font-weight: 700;
}
</style>
