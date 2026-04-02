Git 与 GitHub 多人协作指南

详细教程：https://blog.csdn.net/yeye_queenmoon/article/details/144472289

在多人协作开发中，Git 负责版本控制，GitHub 作为远程仓库平台，帮助团队共享代码、管理分支与合并请求。核心流程包括：分支管理、代码同步、冲突解决与 Pull Request。

1. 初始化与远程关联

git config --global user.name "你的名字"
git config --global user.email "你的邮箱"
git clone <仓库地址> # 克隆远程仓库
git remote -v # 查看远程仓库信息
复制
首次推送需建立追踪关系：

git push -u origin <分支名>
复制
2. 分支协作流程

创建功能分支（避免直接在主分支开发）：

git checkout -b feature/功能描述 main
复制
日常开发：

git add
git commit -m "feat: 完成某功能"
git push origin feature/功能描述
复制
保持分支最新（避免冲突）：

git fetch upstream
git merge upstream/main # 或 git rebase upstream/main
复制
3. 多人协作常见模式

推送失败（远程有新提交）：

git pull origin <分支名> # 拉取并合并
# 若有冲突，手动修改后：
git add 冲突文件
git commit -m "fix: 解决冲突"
git push origin <分支名>
复制
建立本地与远程分支关联：

git branch --set-upstream-to=origin/<远程分支> <本地分支>
复制
4. Pull Request 流程（GitHub）

在本地完成开发并推送到远程功能分支。

打开 GitHub，点击 Compare & pull request。

填写修改说明、关联 Issue，提交 PR。

代码审查通过后合并到主分支。

5. 冲突处理技巧 冲突标记示例：

<<<<<<< HEAD
你的修改
=======
远程修改
>>>>>>> main
复制
解决后：

git add
git commit -m "fix: 合并冲突"
复制
6. 最佳实践

小步提交，每次提交只做一件事。

提交信息规范化（如 feat/fix/docs）。

工作前先同步主分支：git pull origin main。

使用 .gitignore 忽略不必要文件（如 build/、venv/）。

这样，团队即可高效利用 Git + GitHub 进行多人协作，减少冲突并保持代码库整洁。