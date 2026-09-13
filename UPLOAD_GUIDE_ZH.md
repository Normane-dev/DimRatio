# 匿名 GitHub 上传指南

## 1. 上传前准备

1. 使用不包含姓名、学校或常用用户名的新 GitHub 账号。
2. 在该账号的 GitHub 邮箱设置中复制 `noreply` 邮箱地址。
3. 新建一个空的公开仓库，例如 `dimratio-anonymous`；不要自动添加 README、License 或 `.gitignore`。

## 2. 在 PowerShell 中上传

进入本项目目录后依次运行：

```powershell
cd "D:\研究生文件\工业图生成\论文内容准备\DimRatio_Anonymous_GitHub"
git init
git branch -M main
git config user.name "Anonymous Authors"
git config user.email "替换为匿名账号的noreply邮箱"
git add .
git status
git commit -m "Initial anonymous review artifact"
git remote add origin https://github.com/替换为匿名账号/dimratio-anonymous.git
git push -u origin main
```

如果 GitHub 要求登录，请使用匿名账号完成浏览器授权或使用该账号的 personal access token，不要使用个人账号凭据。

## 3. 开启项目页面

在仓库中进入 `Settings → Pages`，将 Source 设为 `Deploy from a branch`，选择 `main` 和 `/docs`，保存后等待部署完成。

页面地址通常是：

```text
https://匿名账号.github.io/dimratio-anonymous/
```

## 4. 最终匿名检查

- GitHub 账号名、头像、邮箱和个人主页不暴露作者身份。
- 提交作者显示为 `Anonymous Authors`。
- README、文件路径、图片元数据和日志中没有姓名、单位、服务器地址或密钥。
- 论文中引用的是匿名仓库/页面链接，且不要通过个人账号 fork、star 或公开互动。
- 双盲期内不要在仓库 issue、release 或 commit message 中透露身份。

