<template>
  <div>
    <div class="train-controls">
      <button class="button-add-example button--green" v-on:click="predict()">{{state}}</button>
      <pre>Prediction : {{prediction}}</pre>
      <pre>Data : {{predictionData}}</pre>
    </div>
  </div>
</template>

<script>
import * as tf from "@tensorflow/tfjs";

export default {
  data() {
    return {
      prediction: 0,
      state: "Predict",
      predictionData: {}
    };
  },
  methods: {
    async predict() {
      this.state = "Thinking ...";

      const model = tf.sequential();
      model.add(tf.layers.dense({ units: 1, inputShape: [1] }));

      model.compile({
        loss: "meanSquaredError",
        optimizer: "sgd"
      });

      const xs = tf.tensor2d([-1, 0, 1, 2, 3, 4], [6, 1]);
      const ys = tf.tensor2d([-3, -1, 1, 3, 5, 7], [6, 1]);

      await model.fit(xs, ys, { epochs: 500 });

      this.predictionData = model.predict(tf.tensor2d([10], [1, 1]));
      this.prediction = this.predictionData.dataSync()[0];
      console.log(this.prediction);
      this.state = "Predict";
    }
  }
};
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
