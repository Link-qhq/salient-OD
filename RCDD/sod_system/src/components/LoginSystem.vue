<!-- <template>
  <div class="login-container">
    <el-card class="box-card">
      <h2 style="text-align: center;">铁路缺陷检测系统</h2>
      <el-form :model="form" label-width="80px" :rules="rules" class="form-container">
        <el-form-item label="用户名" prop="username">
          <el-input v-model="form.username" />
        </el-form-item>
        <el-form-item label="密码" prop="password">
          <el-input v-model="form.password" type="password" />
        </el-form-item>
        <el-button class="footer-btn" type="primary" @click="handleLogin">登录</el-button>
      </el-form>
    </el-card>
  </div>
</template>

<script setup>
import { reactive } from 'vue'
import { useRouter } from 'vue-router'
import axios from 'axios'
import { ElMessage } from 'element-plus'

const router = useRouter()
const form = reactive({
  username: '',
  password: ''
})
const rules = {
  username: [{ required: true, message: '请输入用户名', trigger: 'blur' }],
  password: [{ required: true, message: '请输入密码', trigger: 'blur' }]
}
const handleLogin = async () => {
  try {
    const res = await axios.post('http://localhost:5000/api/login', form)
    localStorage.setItem('token', res.data.token)
    router.push('/')
  } catch (error) {
    ElMessage.error('登录失败')
  }
}
</script>

<style lang="css" scoped>
.login-container {
  width: 600px;
  padding: 20px;
  position: absolute;
  left: 50%;
  top: 30%;
  transform: translate(-50%, -50%);
}
.form-container {
  display: flex;
  flex-direction: column;
}
.footer-btn {
  width: calc(100% - 80px);
  align-self: flex-end;
}
</style> -->

<template>
  <div class="login-container">
    <!-- 新增左侧系统介绍区域 -->
    <div class="left-panel">
      <h1>铁轨表面缺陷检测系统</h1>
      <!-- <div class="features">
        <div v-for="(item,index) in features" :key="index" class="feature-item">
          <el-icon><component :is="item.icon"/></el-icon>
          <span>{{ item.text }}</span>
        </div>
      </div> -->
    </div>

    <!-- 优化右侧登录表单 -->
    <el-card class="login-box">
      <div class="logo">
        <img src="@/assets/images/rail-icon.png" alt="system-logo">
      </div>
      <el-form ref="formRef" :model="form" :rules="rules">
        <el-form-item prop="username">
          <el-input 
            v-model="form.username"
            placeholder="工号/用户名"
            prefix-icon="User"
            size="large"
          />
        </el-form-item>

        <el-form-item prop="password">
          <el-input
            v-model="form.password"
            type="password"
            placeholder="登录密码"
            prefix-icon="Lock"
            size="large"
            show-password
          />
        </el-form-item>

        <el-button 
          type="primary" 
          size="large"
          class="login-btn"
          :loading="loading"
          @click="handleLogin"
        >
          安全登录
        </el-button>
      </el-form>
    </el-card>
  </div>
</template>
<script setup>
import { ref, reactive } from 'vue'
import { useRouter } from 'vue-router'
// import axios from 'axios'
import service from '@/request'
import { ElMessage } from 'element-plus'
// import { 
//   Monitor,
//   Warning,
//   Histogram,
//   DataAnalysis 
// } from '@element-plus/icons-vue'

const router = useRouter()
const form =  reactive({
  username: '',
  password: ''
})
const formRef = ref(null)
// 功能特性数据
// const features = ref([
//   {
//     icon: Monitor,
//     text: '在线检测'
//   },
//   {
//     icon: Warning,
//     text: '智能预警'
//   },
//   {
//     icon: Histogram,
//     text: '精准分析'
//   },
//   {
//     icon: DataAnalysis,
//     text: '数据可视化'
//   }
// ])
// 表单验证规则
const rules = reactive({
  username: [
    { required: true, message: '请输入用户名', trigger: 'blur' },
    { min: 4, max: 16, message: '长度在4到16个字符', trigger: 'blur' }
  ],
  password: [
    { required: true, message: '请输入密码', trigger: 'blur' },
    { pattern: /^(?=.*[A-Za-z])(?=.*\d)[A-Za-z\d]{6,}$/, 
      message: '至少8位字母数字组合' }
  ],
  captcha: [
    { required: true, message: '请输入验证码', trigger: 'blur' },
    { len: 4, message: '验证码为4位字符', trigger: 'blur' }
  ]
})

const handleLogin = () => {
  formRef.value.validate(async (valid) => {
    if (valid) {
      try {
        await service.post('/api/login', form, {
          'headers': {
            "Content-Type": 'application/json'
          }
        })
        router.push({
            path: '/home'
          })
        // const res = await Promise.resolve({
        //   data: {
        //     token: 'xxxx'
        //   }
        // })
        // if (form.username === 'admin' && form.password === '123qwe') {
        //   localStorage.setItem('token', res.data.token)
        //   router.push({
        //     path: '/home'
        //   })
        // }
      } catch (error) {
        ElMessage.error('登录失败')
      }
    }
  })
  
}
</script>
<style lang="scss" scoped>
/* 新增PC端适配样式 */
.login-container {
  display: flex;
  flex-direction: column;
  align-items: center;
  height: calc(100vh - 20px);
  background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);

  .left-panel {
    width: calc(100% - 80px);
    padding: 40px;
    background: linear-gradient(135deg, 
    rgba(0, 40, 77, 0.95) 0%, 
    rgba(0, 72, 144, 0.95) 100%);
    color: white;
    text-align: center;

    h1 {
      font-size: 2.5rem;
      margin-bottom: 2rem;
      letter-spacing: 20px;
    }
    
    .feature-item {
      display: flex;
      align-items: center;
      margin: 1.5rem 0;
      font-size: 1.1rem;
      
      .el-icon {
        font-size: 1.8rem;
        margin-right: 1rem;
      }
    }
  }

  .login-box {
    width: 480px;
    margin: auto 60px;
    border-radius: 10px;
    box-shadow: 0 6px 20px rgba(0,0,0,0.1);
    
    .logo {
      text-align: center;
      margin-bottom: 2rem;
      
      img {
        height: 80px;
      }
    }
    
    .login-btn {
      width: 100%;
      margin-top: 20px;
    }
    
    .captcha-container {
      display: flex;
      gap: 10px;
      
      .captcha-img {
        height: 40px;
        cursor: pointer;
        border-radius: 4px;
      }
    }
    
    .form-options {
      display: flex;
      justify-content: space-between;
      margin: 1rem 0;
    }
  }
}
</style>