<template>
  <!-- <img alt="Vue logo" src="./assets/image.jpg"> -->
  <!-- <HelloWorld msg="Welcome to Your Vue.js App"/> -->
   <!-- <h1>检测平台</h1> -->
   <!-- <HomeHost></HomeHost> -->
   <RouterView></RouterView>
</template>

<script>
// import HelloWorld from './components/HelloWorld.vue'
// import HomeHost from './components/HomeHost.vue';
// import LoginSystem from './components/LoginSystem.vue';
import { RouterView } from 'vue-router'
export default {
  name: 'App',
  components: {
    // HomeHost,
    // LoginSystem
    RouterView
    
  }
}
</script>

<style>
#app {
  font-family: Inter, 'Helvetica Neue', Helvetica, 'PingFang SC',
  'Hiragino Sans GB', 'Microsoft YaHei', '微软雅黑', Arial, sans-serif;
  -webkit-font-smoothing: antialiased;
  -moz-osx-font-smoothing: grayscale;
  /* text-align: center; */
  color: #2c3e50;
  /* margin-top: 60px; */
  /* padding: 20px; */
  margin: 0;
  padding: 0;
  height: 100%;
  width: 100%;
  background-color: #F2F3F5;
  box-sizing: border-box;
}
</style>
