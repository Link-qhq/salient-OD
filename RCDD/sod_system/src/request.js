import axios from 'axios'

const service = axios.create({
  baseURL: 'http://127.0.0.1:5000/', // Flask后端地址
  timeout: 5000,
  headers: {
    'Content-Type': 'application/json;charset=utf-8'
  }
})

// 请求拦截器
service.interceptors.request.use(config => {
  // 可在此处添加token
  return config
})

// 响应拦截器
service.interceptors.response.use(response => {
  return response.data
}, error => {
  return Promise.reject(error)
})

export default service