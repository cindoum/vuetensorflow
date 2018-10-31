<template>
    <div>
        <app-logo/>
      <h1 class="title">
        Vue | Nuxt | Tensorflow
      </h1>
      <h2 class="subtitle">
        <nuxt-link to="/prediction">Prediction</nuxt-link> | 
        <nuxt-link to="/Pair">Pair</nuxt-link> | 
        <nuxt-link to="/mobilenet">MobileNet</nuxt-link>
      </h2>
    </div>
</template>

<script>
import AppLogo from "~/components/AppLogo.vue";

export default {
  components: {
    AppLogo
  }
};
</script>
