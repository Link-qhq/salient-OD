import { createMemoryHistory, createRouter } from 'vue-router'
import HomeHost from './components/HomeHost.vue';
import LoginSystem from './components/LoginSystem.vue';

const routes = [
  { 
    path: '/', 
    redirect: '/login'
  },
  { 
    path: '/login', 
    name: 'LOGIN',
    component: LoginSystem,
    meta: {
      title: '系统登录',
    }
  },
  { 
    path: '/home', 
    name: 'HOME',
    component: HomeHost,
    meta: {
      title: '检测界面',
    } 
  },
]

export const router = createRouter({
  history: createMemoryHistory(),
  routes,
})