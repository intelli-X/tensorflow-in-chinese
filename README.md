Tensorflow in Chinese
=============================================

中文版 Google官方Tensorflow教程

[英文原版](https://github.com/tensorflow/tensorflow)

**如果想帮忙翻译或者校对，请留言，谢谢！**


# 当前阶段

已经翻译 Tensorflow 1.0 官方文档


# 贡献力量

如果想做出贡献的话，你可以：

- 帮忙校对，挑错别字、病句等等
- 提出修改建议
- 提出术语翻译建议

# 翻译建议

如果你愿意一起校对的话，请仔细阅读：

- 使用markdown进行翻译，文件名必须使用英文，因为中文的话gitbook编译的时候会出问题
- 翻译后的文档请放到source文件夹下的对应章节中，然后pull request即可，我会用gitbook编译成网页
- 工作分支为gh-pages，用于GitHub的pages服务
- fork过去之后新建一个分支进行翻译，完成后pull request这个分支，没问题的话我会合并到gh-pages分支中
- 有其他任何问题都欢迎发issue，我看到了会尽快回复

谢谢！

# 关于术语

翻译术语的时候请参考这个流程：

- 尽量保证和已翻译的内容一致
- 如果以上两条都没有找到合适的结果，请自己决定一个合适的翻译或者直接使用英文原文，后期校对的时候会进行统一

# 参考流程

有些朋友可能不太清楚如何帮忙翻译，我这里写一个简单的流程，大家可以参考一下：

1. 首先fork我的项目
2. 把fork过去的项目也就是你的项目clone到你的本地
3. 在命令行运行 `git branch develop` 来创建一个新分支
4. 运行 `git checkout develop` 来切换到新分支
5. 运行 `git remote add upstream https://github.com/intelli-X/tensorflow-in-chinese.git` 把我的库添加为远端库
6. 运行 `git remote update`更新
7. 运行 `git fetch upstream gh-pages` 拉取我的库的更新到本地
8. 运行 `git rebase upstream/gh-pages` 将我的更新合并到你的分支

这是一个初始化流程，只需要做一遍就行，之后请一直在develop分支进行修改。

如果修改过程中我的库有了更新，请重复6、7、8步。

修改之后，首先push到你的库，然后登录GitHub，在你的库的首页可以看到一个 `pull request` 按钮，点击它，填写一些说明信息，然后提交即可。





