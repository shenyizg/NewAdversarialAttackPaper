# Latest Adversarial Attack Papers
**update at 2025-08-24 10:00:57**

翻译来自 https://cloud.tencent.com/document/product/551/15619

## **1. Distributed Detection of Adversarial Attacks in Multi-Agent Reinforcement Learning with Continuous Action Space**

具有连续动作空间的多智能体强化学习中对抗性攻击的分布式检测 cs.LG

Accepted for publication at ECAI 2025

**SubmitDate**: 2025-08-21    [abs](http://arxiv.org/abs/2508.15764v1) [paper-pdf](http://arxiv.org/pdf/2508.15764v1)

**Authors**: Kiarash Kazari, Ezzeldin Shereen, György Dán

**Abstract**: We address the problem of detecting adversarial attacks against cooperative multi-agent reinforcement learning with continuous action space. We propose a decentralized detector that relies solely on the local observations of the agents and makes use of a statistical characterization of the normal behavior of observable agents. The proposed detector utilizes deep neural networks to approximate the normal behavior of agents as parametric multivariate Gaussian distributions. Based on the predicted density functions, we define a normality score and provide a characterization of its mean and variance. This characterization allows us to employ a two-sided CUSUM procedure for detecting deviations of the normality score from its mean, serving as a detector of anomalous behavior in real-time. We evaluate our scheme on various multi-agent PettingZoo benchmarks against different state-of-the-art attack methods, and our results demonstrate the effectiveness of our method in detecting impactful adversarial attacks. Particularly, it outperforms the discrete counterpart by achieving AUC-ROC scores of over 0.95 against the most impactful attacks in all evaluated environments.

摘要: 我们解决了检测针对具有连续动作空间的合作多智能体强化学习的对抗攻击的问题。我们提出了一种去中心化的检测器，它仅依赖于代理的局部观察，并利用可观察代理的正常行为的统计特征。提出的检测器利用深度神经网络将代理的正常行为逼近为参数多元高斯分布。基于预测的密度函数，我们定义正态分数并提供其均值和方差的描述。这种特征使我们能够使用双边CLARUM程序来检测正态分数与其平均值的偏差，作为实时异常行为的检测器。我们在各种多代理PettingZoo基准上针对不同最先进的攻击方法评估了我们的方案，我们的结果证明了我们的方法在检测有影响力的对抗性攻击方面的有效性。特别是，它在所有评估环境中针对最具影响力的攻击时，其AUC-ROC得分超过0.95，优于离散对应的。



## **2. Let's Measure Information Step-by-Step: LLM-Based Evaluation Beyond Vibes**

让我们逐步衡量信息：超越共鸣的基于LLM的评估 cs.LG

Add AUC results, pre-reg conformance, theory section clarification.  12 pages

**SubmitDate**: 2025-08-21    [abs](http://arxiv.org/abs/2508.05469v2) [paper-pdf](http://arxiv.org/pdf/2508.05469v2)

**Authors**: Zachary Robertson, Sanmi Koyejo

**Abstract**: We study evaluation of AI systems without ground truth by exploiting a link between strategic gaming and information loss. We analyze which information-theoretic mechanisms resist adversarial manipulation, extending finite-sample bounds to show that bounded f-divergences (e.g., total variation distance) maintain polynomial guarantees under attacks while unbounded measures (e.g., KL divergence) degrade exponentially. To implement these mechanisms, we model the overseer as an agent and characterize incentive-compatible scoring rules as f-mutual information objectives. Under adversarial attacks, TVD-MI maintains effectiveness (area under curve 0.70-0.77) while traditional judge queries are near change (AUC $\approx$ 0.50), demonstrating that querying the same LLM for information relationships rather than quality judgments provides both theoretical and practical robustness. The mechanisms decompose pairwise evaluations into reliable item-level quality scores without ground truth, addressing a key limitation of traditional peer prediction. We release preregistration and code.

摘要: 我们通过利用战略游戏和信息丢失之间的联系来研究在没有基本真相的情况下对人工智能系统的评估。我们分析哪些信息论机制抵抗对抗操纵，扩展有限样本界限以表明有界f-分歧（例如，总变化距离）在攻击下保持多项保证，而无界措施（例如，KL分歧）呈指数级下降。为了实现这些机制，我们将监督者建模为代理，并将激励兼容评分规则描述为f-互信息目标。在对抗性攻击下，TVD-MI保持有效性（曲线下面积0.70-0.77），而传统的判断查询几乎发生变化（UC $\大约$0.50），这表明查询相同的LLM以获取信息关系而不是质量判断提供了理论和实践的鲁棒性。这些机制将成对评估分解为可靠的项目级质量分数，无需基本真相，解决了传统同伴预测的关键局限性。我们发布预注册和代码。



## **3. Towards a 3D Transfer-based Black-box Attack via Critical Feature Guidance**

通过关键功能指导进行基于3D传输的黑匣子攻击 cs.CV

11 pages, 6 figures

**SubmitDate**: 2025-08-21    [abs](http://arxiv.org/abs/2508.15650v1) [paper-pdf](http://arxiv.org/pdf/2508.15650v1)

**Authors**: Shuchao Pang, Zhenghan Chen, Shen Zhang, Liming Lu, Siyuan Liang, Anan Du, Yongbin Zhou

**Abstract**: Deep neural networks for 3D point clouds have been demonstrated to be vulnerable to adversarial examples. Previous 3D adversarial attack methods often exploit certain information about the target models, such as model parameters or outputs, to generate adversarial point clouds. However, in realistic scenarios, it is challenging to obtain any information about the target models under conditions of absolute security. Therefore, we focus on transfer-based attacks, where generating adversarial point clouds does not require any information about the target models. Based on our observation that the critical features used for point cloud classification are consistent across different DNN architectures, we propose CFG, a novel transfer-based black-box attack method that improves the transferability of adversarial point clouds via the proposed Critical Feature Guidance. Specifically, our method regularizes the search of adversarial point clouds by computing the importance of the extracted features, prioritizing the corruption of critical features that are likely to be adopted by diverse architectures. Further, we explicitly constrain the maximum deviation extent of the generated adversarial point clouds in the loss function to ensure their imperceptibility. Extensive experiments conducted on the ModelNet40 and ScanObjectNN benchmark datasets demonstrate that the proposed CFG outperforms the state-of-the-art attack methods by a large margin.

摘要: 用于3D点云的深度神经网络已被证明容易受到对抗示例的影响。之前的3D对抗攻击方法通常利用有关目标模型的某些信息（例如模型参数或输出）来生成对抗点云。然而，在现实场景中，在绝对安全的条件下获取有关目标模型的任何信息是具有挑战性的。因此，我们专注于基于传输的攻击，其中生成对抗点云不需要有关目标模型的任何信息。根据我们观察到用于点云分类的关键特征在不同的DNN架构中是一致的，我们提出了CGM，这是一种新型的基于传输的黑匣子攻击方法，通过提出的关键特征指南提高了对抗性点云的可传输性。具体来说，我们的方法通过计算提取特征的重要性、优先考虑可能被不同架构采用的关键特征的损坏，来规范对抗性点云的搜索。此外，我们在损失函数中明确限制生成的对抗点云的最大偏差程度，以确保其不可感知性。在Model Net 40和ScanObjectNN基准数据集上进行的大量实验表明，拟议的CGM的性能大大优于最先进的攻击方法。



## **4. Vulnerabilities in AI-generated Image Detection: The Challenge of Adversarial Attacks**

人工智能生成图像检测中的漏洞：对抗性攻击的挑战 cs.CV

**SubmitDate**: 2025-08-21    [abs](http://arxiv.org/abs/2407.20836v4) [paper-pdf](http://arxiv.org/pdf/2407.20836v4)

**Authors**: Yunfeng Diao, Naixin Zhai, Changtao Miao, Zitong Yu, Xingxing Wei, Xun Yang, Meng Wang

**Abstract**: Recent advancements in image synthesis, particularly with the advent of GAN and Diffusion models, have amplified public concerns regarding the dissemination of disinformation. To address such concerns, numerous AI-generated Image (AIGI) Detectors have been proposed and achieved promising performance in identifying fake images. However, there still lacks a systematic understanding of the adversarial robustness of AIGI detectors. In this paper, we examine the vulnerability of state-of-the-art AIGI detectors against adversarial attack under white-box and black-box settings, which has been rarely investigated so far. To this end, we propose a new method to attack AIGI detectors. First, inspired by the obvious difference between real images and fake images in the frequency domain, we add perturbations under the frequency domain to push the image away from its original frequency distribution. Second, we explore the full posterior distribution of the surrogate model to further narrow this gap between heterogeneous AIGI detectors, e.g. transferring adversarial examples across CNNs and ViTs. This is achieved by introducing a novel post-train Bayesian strategy that turns a single surrogate into a Bayesian one, capable of simulating diverse victim models using one pre-trained surrogate, without the need for re-training. We name our method as Frequency-based Post-train Bayesian Attack, or FPBA. Through FPBA, we show that adversarial attack is truly a real threat to AIGI detectors, because FPBA can deliver successful black-box attacks across models, generators, defense methods, and even evade cross-generator detection, which is a crucial real-world detection scenario. The code will be shared upon acceptance.

摘要: 图像合成领域的最新进展，特别是GAN和扩散模型的出现，加剧了公众对虚假信息传播的担忧。为了解决这些问题，人们提出了许多人工智能生成的图像（AIGI）检测器，并在识别虚假图像方面取得了良好的性能。然而，人们仍然缺乏对AIGI检测器对抗鲁棒性的系统了解。在本文中，我们研究了最先进的AIGI检测器在白盒和黑盒设置下对抗攻击的脆弱性，迄今为止，这很少被研究。为此，我们提出了一种攻击AIGI检测器的新方法。首先，受到频域中真实图像和假图像之间明显差异的启发，我们在频域下添加扰动，以推动图像远离其原始频率分布。其次，我们探索代理模型的完整后验分布，以进一步缩小异类AIGI检测器之间的差距，例如跨CNN和ViT传输对抗性示例。这是通过引入一种新型的训练后Bayesian策略来实现的，该策略将单个代理变成了Bayesian代理，能够使用一个预先训练的代理来模拟不同的受害者模型，而无需重新训练。我们将我们的方法命名为基于频率的训练后Bayesian Attack（FPBA）。通过FPBA，我们表明对抗性攻击确实是对AIGI检测器的真正威胁，因为FPBA可以跨模型、生成器、防御方法提供成功的黑匣子攻击，甚至逃避交叉生成器检测，这是一个至关重要的现实世界检测场景。该代码将在接受后共享。



## **5. Any-to-any Speaker Attribute Perturbation for Asynchronous Voice Anonymization**

基于Any-to-Any说话人属性扰动的异步语音识别 cs.SD

**SubmitDate**: 2025-08-21    [abs](http://arxiv.org/abs/2508.15565v1) [paper-pdf](http://arxiv.org/pdf/2508.15565v1)

**Authors**: Liping Chen, Chenyang Guo, Rui Wang, Kong Aik Lee, Zhenhua Ling

**Abstract**: Speaker attribute perturbation offers a feasible approach to asynchronous voice anonymization by employing adversarially perturbed speech as anonymized output. In order to enhance the identity unlinkability among anonymized utterances from the same original speaker, the targeted attack training strategy is usually applied to anonymize the utterances to a common designated speaker. However, this strategy may violate the privacy of the designated speaker who is an actual speaker. To mitigate this risk, this paper proposes an any-to-any training strategy. It is accomplished by defining a batch mean loss to anonymize the utterances from various speakers within a training mini-batch to a common pseudo-speaker, which is approximated as the average speaker in the mini-batch. Based on this, a speaker-adversarial speech generation model is proposed, incorporating the supervision from both the untargeted attack and the any-to-any strategies. The speaker attribute perturbations are generated and incorporated into the original speech to produce its anonymized version. The effectiveness of the proposed model was justified in asynchronous voice anonymization through experiments conducted on the VoxCeleb datasets. Additional experiments were carried out to explore the potential limitations of speaker-adversarial speech in voice privacy protection. With them, we aim to provide insights for future research on its protective efficacy against black-box speaker extractors \textcolor{black}{and adaptive attacks, as well as} generalization to out-of-domain datasets \textcolor{black}{and stability}. Audio samples and open-source code are published in https://github.com/VoicePrivacy/any-to-any-speaker-attribute-perturbation.

摘要: 说话者属性扰动通过使用对抗扰动的语音作为匿名输出，为同步语音匿名化提供了一种可行的方法。为了增强来自同一原始说话者的匿名话语之间的身份不可链接性，通常应用有针对性的攻击训练策略来匿名化共同指定说话者的话语。然而，这种策略可能会侵犯指定发言人（实际发言人）的隐私。为了降低这一风险，本文提出了一种任意对任意的培训策略。它是通过定义批量平均损失来实现的，以将训练小批量内各个说话者的话语匿名到常见的伪说话者，该伪说话者被逼近为小批量中的平均说话者。在此基础上，提出了一种说话者对抗语音生成模型，该模型结合了无目标攻击和任何对任何策略的监督。生成说话者属性扰动并将其合并到原始语音中以产生其匿名版本。通过在VoxCeleb数据集上进行的实验，证明了所提出模型在同步语音匿名化中的有效性。还进行了额外的实验来探索说话者对抗性言语在语音隐私保护中的潜在局限性。通过它们，我们的目标是为未来的研究提供见解，了解其针对黑匣子扬声器提取器\textColor{black}{和自适应攻击的保护功效，以及}对域外数据集\textColor{black}{和稳定性}的概括。音频样本和开源代码发布在https://github.com/VoicePrivacy/any-to-any-speaker-attribute-perturbation上。



## **6. BadFU: Backdoor Federated Learning through Adversarial Machine Unlearning**

BadFU：通过对抗机器取消学习的后门联邦学习 cs.CR

**SubmitDate**: 2025-08-21    [abs](http://arxiv.org/abs/2508.15541v1) [paper-pdf](http://arxiv.org/pdf/2508.15541v1)

**Authors**: Bingguang Lu, Hongsheng Hu, Yuantian Miao, Shaleeza Sohail, Chaoxiang He, Shuo Wang, Xiao Chen

**Abstract**: Federated learning (FL) has been widely adopted as a decentralized training paradigm that enables multiple clients to collaboratively learn a shared model without exposing their local data. As concerns over data privacy and regulatory compliance grow, machine unlearning, which aims to remove the influence of specific data from trained models, has become increasingly important in the federated setting to meet legal, ethical, or user-driven demands. However, integrating unlearning into FL introduces new challenges and raises largely unexplored security risks. In particular, adversaries may exploit the unlearning process to compromise the integrity of the global model. In this paper, we present the first backdoor attack in the context of federated unlearning, demonstrating that an adversary can inject backdoors into the global model through seemingly legitimate unlearning requests. Specifically, we propose BadFU, an attack strategy where a malicious client uses both backdoor and camouflage samples to train the global model normally during the federated training process. Once the client requests unlearning of the camouflage samples, the global model transitions into a backdoored state. Extensive experiments under various FL frameworks and unlearning strategies validate the effectiveness of BadFU, revealing a critical vulnerability in current federated unlearning practices and underscoring the urgent need for more secure and robust federated unlearning mechanisms.

摘要: 联合学习（FL）已被广泛采用作为一种去中心化的培训范式，使多个客户能够协作学习共享模型，而无需暴露其本地数据。随着人们对数据隐私和监管合规性的担忧日益加剧，旨在消除训练模型中特定数据影响的机器去学习在联邦环境中变得越来越重要，以满足法律、道德或用户驱动的需求。然而，将取消学习集成到FL中带来了新的挑战，并引发了基本上未探索的安全风险。特别是，对手可能会利用取消学习过程来损害全球模型的完整性。在本文中，我们在联邦取消学习的背景下提出了第一次后门攻击，证明对手可以通过看似合法的取消学习请求将后门注入全局模型。具体来说，我们提出了BadFU，这是一种攻击策略，恶意客户端使用后门和伪装样本来在联邦训练过程中正常训练全局模型。一旦客户端请求取消伪装样本的学习，全局模型就会转变为后门状态。各种FL框架和取消学习策略下的广泛实验验证了BadFU的有效性，揭示了当前联邦取消学习实践中的一个关键漏洞，并强调了对更安全和更强大的联邦取消学习机制的迫切需求。



## **7. On Evaluating the Adversarial Robustness of Foundation Models for Multimodal Entity Linking**

多模式实体链接基础模型的对抗鲁棒性评估 cs.IR

**SubmitDate**: 2025-08-21    [abs](http://arxiv.org/abs/2508.15481v1) [paper-pdf](http://arxiv.org/pdf/2508.15481v1)

**Authors**: Fang Wang, Yongjie Wang, Zonghao Yang, Minghao Hu, Xiaoying Bai

**Abstract**: The explosive growth of multimodal data has driven the rapid development of multimodal entity linking (MEL) models. However, existing studies have not systematically investigated the impact of visual adversarial attacks on MEL models. We conduct the first comprehensive evaluation of the robustness of mainstream MEL models under different adversarial attack scenarios, covering two core tasks: Image-to-Text (I2T) and Image+Text-to-Text (IT2T). Experimental results show that current MEL models generally lack sufficient robustness against visual perturbations. Interestingly, contextual semantic information in input can partially mitigate the impact of adversarial perturbations. Based on this insight, we propose an LLM and Retrieval-Augmented Entity Linking (LLM-RetLink), which significantly improves the model's anti-interference ability through a two-stage process: first, extracting initial entity descriptions using large vision models (LVMs), and then dynamically generating candidate descriptive sentences via web-based retrieval. Experiments on five datasets demonstrate that LLM-RetLink improves the accuracy of MEL by 0.4%-35.7%, especially showing significant advantages under adversarial conditions. This research highlights a previously unexplored facet of MEL robustness, constructs and releases the first MEL adversarial example dataset, and sets the stage for future work aimed at strengthening the resilience of multimodal systems in adversarial environments.

摘要: 多模式数据的爆炸式增长推动了多模式实体链接（MEL）模型的快速发展。然而，现有的研究尚未系统地研究视觉对抗攻击对MEL模型的影响。我们对主流MEL模型在不同对抗攻击场景下的稳健性进行了首次全面评估，涵盖两项核心任务：图像到文本（I2 T）和图像+文本到文本（IT 2 T）。实验结果表明，当前的MEL模型对于视觉扰动通常缺乏足够的鲁棒性。有趣的是，输入中的上下文语义信息可以部分减轻对抗性扰动的影响。基于这一认识，我们提出了一个LLM和检索增强实体链接（LLM-RetLink），它显着提高了模型的抗干扰能力，通过两个阶段的过程：首先，提取初始实体描述使用大型视觉模型（LVMs），然后通过基于Web的检索动态生成候选描述语句。在五个数据集上的实验表明，LLM-RetLink将MEL的准确率提高了0.4%-35.7%，特别是在对抗性条件下显示出显著的优势。这项研究突出了MEL鲁棒性的一个以前未被探索的方面，构建并发布了第一个MEL对抗性示例数据集，并为未来旨在加强对抗性环境中多模式系统的弹性的工作奠定了基础。



## **8. Mini-Batch Robustness Verification of Deep Neural Networks**

深度神经网络的小批量鲁棒性验证 cs.LG

30 pages, 12 figures, conference OOPSLA 2025

**SubmitDate**: 2025-08-21    [abs](http://arxiv.org/abs/2508.15454v1) [paper-pdf](http://arxiv.org/pdf/2508.15454v1)

**Authors**: Saar Tzour-Shaday, Dana Drachsler Cohen

**Abstract**: Neural network image classifiers are ubiquitous in many safety-critical applications. However, they are susceptible to adversarial attacks. To understand their robustness to attacks, many local robustness verifiers have been proposed to analyze $\epsilon$-balls of inputs. Yet, existing verifiers introduce a long analysis time or lose too much precision, making them less effective for a large set of inputs. In this work, we propose a new approach to local robustness: group local robustness verification. The key idea is to leverage the similarity of the network computations of certain $\epsilon$-balls to reduce the overall analysis time. We propose BaVerLy, a sound and complete verifier that boosts the local robustness verification of a set of $\epsilon$-balls by dynamically constructing and verifying mini-batches. BaVerLy adaptively identifies successful mini-batch sizes, accordingly constructs mini-batches of $\epsilon$-balls that have similar network computations, and verifies them jointly. If a mini-batch is verified, all $\epsilon$-balls are proven robust. Otherwise, one $\epsilon$-ball is suspected as not being robust, guiding the refinement. In the latter case, BaVerLy leverages the analysis results to expedite the analysis of that $\epsilon$-ball as well as the other $\epsilon$-balls in the batch. We evaluate BaVerLy on fully connected and convolutional networks for MNIST and CIFAR-10. Results show that BaVerLy scales the common one by one verification by 2.3x on average and up to 4.1x, in which case it reduces the total analysis time from 24 hours to 6 hours.

摘要: 神经网络图像分类器在许多安全关键应用中无处不在。然而，他们很容易受到敌对攻击。为了了解它们对攻击的鲁棒性，人们提出了许多本地鲁棒性验证器来分析$\$-输入球。然而，现有的验证器引入了较长的分析时间或失去了太多的精确度，使它们对大量输入的效率较低。在这项工作中，我们提出了一种新的局部鲁棒性方法：组局部鲁棒性验证。关键想法是利用某些$\$-balls的网络计算的相似性来减少总体分析时间。我们提出了BaVerLy，这是一个可靠且完整的验证器，通过动态构建和验证迷你批处理来增强一组$\$-balls的局部鲁棒性验证。BaVerLy自适应地识别成功的迷你批量大小，相应地构建具有类似网络计算的$\$-balls的迷你批量，并联合验证它们。如果迷你批次得到验证，则所有$\$-球都被证明是稳健的。否则，一个$\$-ball被怀疑不稳健，从而指导细化。在后一种情况下，BaVerLy利用分析结果来加速对该$\$-ball以及批次中其他$\$-ball的分析。我们在MNIST和CIFAR-10的全连接和卷积网络上评估了BaVerLy。结果显示，BaVerLy将常见的逐个验证平均扩展了2.3倍，最多扩展了4.1倍，在这种情况下，总分析时间从24小时减少到6小时。



## **9. Tensor Train Decomposition for Adversarial Attacks on Computer Vision Models**

计算机视觉模型对抗攻击的张量序列分解 math.NA

**SubmitDate**: 2025-08-21    [abs](http://arxiv.org/abs/2312.12556v2) [paper-pdf](http://arxiv.org/pdf/2312.12556v2)

**Authors**: Andrei Chertkov, Ivan Oseledets

**Abstract**: Deep neural networks (DNNs) are widely used today, but they are vulnerable to adversarial attacks. To develop effective methods of defense, it is important to understand the potential weak spots of DNNs. Often attacks are organized taking into account the architecture of models (white-box approach) and based on gradient methods, but for real-world DNNs this approach in most cases is impossible. At the same time, several gradient-free optimization algorithms are used to attack black-box models. However, classical methods are often ineffective in the multidimensional case. To organize black-box attacks for computer vision models, in this work, we propose the use of an optimizer based on the low-rank tensor train (TT) format, which has gained popularity in various practical multidimensional applications in recent years. Combined with the attribution of the target image, which is built by the auxiliary (white-box) model, the TT-based optimization method makes it possible to organize an effective black-box attack by small perturbation of pixels in the target image. The superiority of the proposed approach over three popular baselines is demonstrated for seven modern DNNs on the ImageNet dataset.

摘要: 深度神经网络（DNN）如今被广泛使用，但它们很容易受到对抗攻击。为了开发有效的防御方法，了解DNN的潜在弱点非常重要。通常，攻击是在考虑模型架构（白盒方法）并基于梯度方法的情况下组织的，但对于现实世界的DNN来说，这种方法在大多数情况下是不可能的。同时，使用多种无梯度优化算法来攻击黑匣子模型。然而，经典方法在多维情况下往往无效。为了组织对计算机视觉模型的黑匣子攻击，在这项工作中，我们提议使用基于低等级张量训练（TT）格式的优化器，该格式近年来在各种实际多维应用中越来越受欢迎。结合辅助（白盒）模型构建的目标图像的属性，基于TT的优化方法使得通过目标图像中像素的微小扰动组织有效的黑匣子攻击成为可能。ImageNet数据集中的七个现代DNN证明了所提出的方法相对于三个流行基线的优越性。



## **10. Setup Once, Secure Always: A Single-Setup Secure Federated Learning Aggregation Protocol with Forward and Backward Secrecy for Dynamic Users**

一次设置，始终安全：针对动态用户的单次设置安全联邦学习聚合协议，具有前向和后向保密性 cs.CR

17 pages, 12 Figures

**SubmitDate**: 2025-08-21    [abs](http://arxiv.org/abs/2502.08989v4) [paper-pdf](http://arxiv.org/pdf/2502.08989v4)

**Authors**: Nazatul Haque Sultan, Yan Bo, Yansong Gao, Seyit Camtepe, Arash Mahboubi, Hang Thanh Bui, Aufeef Chauhan, Hamed Aboutorab, Michael Bewong, Dineshkumar Singh, Praveen Gauravaram, Rafiqul Islam, Sharif Abuadbba

**Abstract**: Federated Learning (FL) enables multiple users to collaboratively train a machine learning model without sharing raw data, making it suitable for privacy-sensitive applications. However, local model or weight updates can still leak sensitive information. Secure aggregation protocols mitigate this risk by ensuring that only the aggregated updates are revealed. Among these, single-setup protocols, where key generation and exchange occur only once, are the most efficient due to reduced communication and computation overhead. However, existing single-setup protocols often lack support for dynamic user participation and do not provide strong privacy guarantees such as forward and backward secrecy. \par In this paper, we present a novel secure aggregation protocol that requires only a single setup for the entire FL training. Our protocol supports dynamic user participation, tolerates dropouts, and achieves both forward and backward secrecy. It leverages lightweight symmetric homomorphic encryption with a key negation technique to mask updates efficiently, eliminating the need for user-to-user communication. To defend against model inconsistency attacks, we introduce a low-overhead verification mechanism using message authentication codes (MACs). We provide formal security proofs under both semi-honest and malicious adversarial models and implement a full prototype. Experimental results show that our protocol reduces user-side computation by up to $99\%$ compared to state-of-the-art protocols like e-SeaFL (ACSAC'24), while maintaining competitive model accuracy. These features make our protocol highly practical for real-world FL deployments, especially on resource-constrained devices.

摘要: 联合学习（FL）使多个用户能够协作训练机器学习模型，而无需共享原始数据，使其适合隐私敏感的应用程序。然而，本地模型或权重更新仍然可能泄露敏感信息。安全聚合协议通过确保仅披露聚合的更新来减轻此风险。其中，密钥生成和交换只发生一次的单次设置协议由于减少了通信和计算负担而是最有效的。然而，现有的单一设置协议往往缺乏对动态用户参与的支持，并且不提供强有力的隐私保证，如前向和后向保密。\par在本文中，我们提出了一种新的安全聚合协议，只需要一个单一的设置为整个FL训练。我们的协议支持动态用户参与，容忍辍学，并实现了前向和后向保密。它利用轻量级对称同态加密和密钥否定技术来有效地屏蔽更新，从而消除了用户对用户通信的需求。为了抵御模型不一致攻击，我们引入了使用消息认证码（MAC）的低负担验证机制。我们在半诚实和恶意对抗模型下提供正式的安全证明，并实现完整的原型。实验结果表明，与e-SeaFL（ACSAC ' 24）等最先进的协议相比，我们的协议将用户端计算减少了高达99美元，同时保持了有竞争力的模型准确性。这些功能使我们的协议对于现实世界的FL部署非常实用，尤其是在资源有限的设备上。



## **11. Adversarial Attacks against Neural Ranking Models via In-Context Learning**

通过上下文学习对神经排名模型的对抗攻击 cs.IR

**SubmitDate**: 2025-08-21    [abs](http://arxiv.org/abs/2508.15283v1) [paper-pdf](http://arxiv.org/pdf/2508.15283v1)

**Authors**: Amin Bigdeli, Negar Arabzadeh, Ebrahim Bagheri, Charles L. A. Clarke

**Abstract**: While neural ranking models (NRMs) have shown high effectiveness, they remain susceptible to adversarial manipulation. In this work, we introduce Few-Shot Adversarial Prompting (FSAP), a novel black-box attack framework that leverages the in-context learning capabilities of Large Language Models (LLMs) to generate high-ranking adversarial documents. Unlike previous approaches that rely on token-level perturbations or manual rewriting of existing documents, FSAP formulates adversarial attacks entirely through few-shot prompting, requiring no gradient access or internal model instrumentation. By conditioning the LLM on a small support set of previously observed harmful examples, FSAP synthesizes grammatically fluent and topically coherent documents that subtly embed false or misleading information and rank competitively against authentic content. We instantiate FSAP in two modes: FSAP-IntraQ, which leverages harmful examples from the same query to enhance topic fidelity, and FSAP-InterQ, which enables broader generalization by transferring adversarial patterns across unrelated queries. Our experiments on the TREC 2020 and 2021 Health Misinformation Tracks, using four diverse neural ranking models, reveal that FSAP-generated documents consistently outrank credible, factually accurate documents. Furthermore, our analysis demonstrates that these adversarial outputs exhibit strong stance alignment and low detectability, posing a realistic and scalable threat to neural retrieval systems. FSAP also effectively generalizes across both proprietary and open-source LLMs.

摘要: 虽然神经排名模型（NRM）表现出很高的有效性，但它们仍然容易受到对抗性操纵的影响。在这项工作中，我们引入了少镜头对抗性过滤（FSAP），这是一种新型的黑盒攻击框架，它利用大型语言模型（LLM）的上下文学习能力来生成高级对抗性文档。与以前依赖于令牌级扰动或手动重写现有文档的方法不同，FSAP完全通过少量提示来制定对抗性攻击，不需要梯度访问或内部模型工具。通过在以前观察到的有害示例的小支持集上调节LLM，FSAP合成了语法流畅和主题连贯的文档，这些文档巧妙地嵌入了虚假或误导性信息，并与真实内容竞争。我们以两种模式实例化FSAP：FSAP-IntraQ，它利用同一查询中的有害示例来增强主题保真度，而FSAP-InterQ，它通过在不相关的查询之间转移对抗模式来实现更广泛的概括。我们使用四种不同的神经排名模型对TREC 2020和2021健康错误信息追踪进行的实验表明，FSAP生成的文档的级别始终高于可信、事实准确的文档。此外，我们的分析表明，这些对抗性输出表现出强的立场对齐和低的可检测性，对神经检索系统构成现实且可扩展的威胁。FSAP还有效地推广了专有和开源LLM。



## **12. SafeLLM: Unlearning Harmful Outputs from Large Language Models against Jailbreak Attacks**

SafeLLM：消除大型语言模型的有害输出以应对越狱攻击 cs.LG

**SubmitDate**: 2025-08-21    [abs](http://arxiv.org/abs/2508.15182v1) [paper-pdf](http://arxiv.org/pdf/2508.15182v1)

**Authors**: Xiangman Li, Xiaodong Wu, Qi Li, Jianbing Ni, Rongxing Lu

**Abstract**: Jailbreak attacks pose a serious threat to the safety of Large Language Models (LLMs) by crafting adversarial prompts that bypass alignment mechanisms, causing the models to produce harmful, restricted, or biased content. In this paper, we propose SafeLLM, a novel unlearning-based defense framework that unlearn the harmful knowledge from LLMs while preserving linguistic fluency and general capabilities. SafeLLM employs a three-stage pipeline: (1) dynamic unsafe output detection using a hybrid approach that integrates external classifiers with model-internal evaluations; (2) token-level harmful content tracing through feedforward network (FFN) activations to localize harmful knowledge; and (3) constrained optimization to suppress unsafe behavior without degrading overall model quality. SafeLLM achieves targeted and irreversible forgetting by identifying and neutralizing FFN substructures responsible for harmful generation pathways. Extensive experiments on prominent LLMs (Vicuna, LLaMA, and GPT-J) across multiple jailbreak benchmarks show that SafeLLM substantially reduces attack success rates while maintaining high general-purpose performance. Compared to standard defense methods such as supervised fine-tuning and direct preference optimization, SafeLLM offers stronger safety guarantees, more precise control over harmful behavior, and greater robustness to unseen attacks. Moreover, SafeLLM maintains the general performance after the harmful knowledge unlearned. These results highlight unlearning as a promising direction for scalable and effective LLM safety.

摘要: 越狱攻击通过精心设计绕过对齐机制的对抗提示，导致模型产生有害、受限制或有偏见的内容，对大型语言模型（LLM）的安全构成严重威胁。在本文中，我们提出了SafeLLM，这是一种新型的基于学习的防御框架，可以从LLM中学习有害知识，同时保持语言流畅性和通用能力。SafeLLM采用三阶段管道：（1）使用将外部分类器与模型内部评估集成的混合方法进行动态不安全输出检测;（2）通过前向网络（FFN）激活进行标记级有害内容跟踪以本地化有害知识;（3）约束优化以抑制不安全行为而不降低整体模型质量。SafeLLM通过识别和中和导致有害生成途径的FFN子结构来实现有针对性且不可逆转的遗忘。在多个越狱基准测试中对知名LLM（Vicuna、LLaMA和GPT-J）进行的广泛实验表明，SafeLLM在保持高通用性能的同时大幅降低了攻击成功率。与监督式微调和直接偏好优化等标准防御方法相比，SafeLLM提供更强的安全保证、对有害行为的更精确控制以及对不可见攻击的更强鲁棒性。此外，SafeLLM在有害知识被遗忘后保持了总体性能。这些结果凸显了取消学习是可扩展和有效的LLM安全性的一个有前途的方向。



## **13. MoEcho: Exploiting Side-Channel Attacks to Compromise User Privacy in Mixture-of-Experts LLMs**

MoEcho：在混合专家LLM中利用侧频道攻击来损害用户隐私 cs.CR

This paper will appear in CCS 2025

**SubmitDate**: 2025-08-20    [abs](http://arxiv.org/abs/2508.15036v1) [paper-pdf](http://arxiv.org/pdf/2508.15036v1)

**Authors**: Ruyi Ding, Tianhong Xu, Xinyi Shen, Aidong Adam Ding, Yunsi Fei

**Abstract**: The transformer architecture has become a cornerstone of modern AI, fueling remarkable progress across applications in natural language processing, computer vision, and multimodal learning. As these models continue to scale explosively for performance, implementation efficiency remains a critical challenge. Mixture of Experts (MoE) architectures, selectively activating specialized subnetworks (experts), offer a unique balance between model accuracy and computational cost. However, the adaptive routing in MoE architectures, where input tokens are dynamically directed to specialized experts based on their semantic meaning inadvertently opens up a new attack surface for privacy breaches. These input-dependent activation patterns leave distinctive temporal and spatial traces in hardware execution, which adversaries could exploit to deduce sensitive user data. In this work, we propose MoEcho, discovering a side channel analysis based attack surface that compromises user privacy on MoE based systems. Specifically, in MoEcho, we introduce four novel architectural side channels on different computing platforms, including Cache Occupancy Channels and Pageout+Reload on CPUs, and Performance Counter and TLB Evict+Reload on GPUs, respectively. Exploiting these vulnerabilities, we propose four attacks that effectively breach user privacy in large language models (LLMs) and vision language models (VLMs) based on MoE architectures: Prompt Inference Attack, Response Reconstruction Attack, Visual Inference Attack, and Visual Reconstruction Attack. MoEcho is the first runtime architecture level security analysis of the popular MoE structure common in modern transformers, highlighting a serious security and privacy threat and calling for effective and timely safeguards when harnessing MoE based models for developing efficient large scale AI services.

摘要: Transformer架构已成为现代人工智能的基石，推动了自然语言处理、计算机视觉和多模式学习等应用的显着进展。随着这些模型的性能持续爆炸式扩展，实施效率仍然是一个关键挑战。混合专家（MoE）架构选择性地激活专业子网络（专家），在模型准确性和计算成本之间提供了独特的平衡。然而，MoE架构中的自适应路由（输入令牌根据其语义动态地引导给专业专家）无意中为隐私泄露开辟了新的攻击面。这些依赖于输入的激活模式在硬件执行中留下独特的时间和空间痕迹，对手可以利用这些痕迹来推断敏感的用户数据。在这项工作中，我们提出了MoEcho，发现一个侧通道分析为基础的攻击面，损害用户隐私的MoE为基础的系统。具体来说，在MoEcho中，我们在不同计算平台上引入了四种新颖的架构侧通道，分别包括处理器上的缓存占用通道和Pageout+ Inbox，以及处理器上的Performance Counter和TSB Evitch + Inbox。利用这些漏洞，我们提出了四种攻击，有效地侵犯用户隐私的大型语言模型（LLM）和视觉语言模型（VLM）的基础上MoE架构：提示推理攻击，响应重建攻击，视觉推理攻击，视觉重建攻击。MoEcho是对现代变压器中常见的流行MoE结构的第一个运行时架构级安全分析，强调了严重的安全和隐私威胁，并呼吁在利用基于MoE的模型开发高效的大规模人工智能服务时采取有效且及时的保障措施。



## **14. A Systematic Survey of Model Extraction Attacks and Defenses: State-of-the-Art and Perspectives**

模型提取攻击和防御的系统性调查：最新技术水平和观点 cs.CR

**SubmitDate**: 2025-08-20    [abs](http://arxiv.org/abs/2508.15031v1) [paper-pdf](http://arxiv.org/pdf/2508.15031v1)

**Authors**: Kaixiang Zhao, Lincan Li, Kaize Ding, Neil Zhenqiang Gong, Yue Zhao, Yushun Dong

**Abstract**: Machine learning (ML) models have significantly grown in complexity and utility, driving advances across multiple domains. However, substantial computational resources and specialized expertise have historically restricted their wide adoption. Machine-Learning-as-a-Service (MLaaS) platforms have addressed these barriers by providing scalable, convenient, and affordable access to sophisticated ML models through user-friendly APIs. While this accessibility promotes widespread use of advanced ML capabilities, it also introduces vulnerabilities exploited through Model Extraction Attacks (MEAs). Recent studies have demonstrated that adversaries can systematically replicate a target model's functionality by interacting with publicly exposed interfaces, posing threats to intellectual property, privacy, and system security. In this paper, we offer a comprehensive survey of MEAs and corresponding defense strategies. We propose a novel taxonomy that classifies MEAs according to attack mechanisms, defense approaches, and computing environments. Our analysis covers various attack techniques, evaluates their effectiveness, and highlights challenges faced by existing defenses, particularly the critical trade-off between preserving model utility and ensuring security. We further assess MEAs within different computing paradigms and discuss their technical, ethical, legal, and societal implications, along with promising directions for future research. This systematic survey aims to serve as a valuable reference for researchers, practitioners, and policymakers engaged in AI security and privacy. Additionally, we maintain an online repository continuously updated with related literature at https://github.com/kzhao5/ModelExtractionPapers.

摘要: 机器学习（ML）模型的复杂性和实用性显着增长，推动了多个领域的进步。然而，大量的计算资源和专业知识历来限制了它们的广泛采用。机器学习即服务（MLaaz）平台通过用户友好的API提供对复杂ML模型的可扩展、方便且经济实惠的访问，从而解决了这些障碍。虽然这种可访问性促进了高级ML功能的广泛使用，但它也引入了通过模型提取攻击（MEAs）利用的漏洞。最近的研究表明，对手可以通过与公开的界面交互来系统性地复制目标模型的功能，从而对知识产权、隐私和系统安全构成威胁。在本文中，我们对多边环境协定和相应的防御策略进行了全面的调查。我们提出了一种新颖的分类法，根据攻击机制、防御方法和计算环境对多边环境进行分类。我们的分析涵盖了各种攻击技术，评估了它们的有效性，并强调了现有防御所面临的挑战，特别是保留模型实用性和确保安全性之间的关键权衡。我们进一步评估不同计算范式中的多边环境协定，并讨论其技术、道德、法律和社会影响，以及未来研究的有希望的方向。这项系统性调查旨在为从事人工智能安全和隐私的研究人员、从业者和政策制定者提供宝贵的参考。此外，我们还在https://github.com/kzhao5/ModelExtractionPapers上维护了一个在线知识库，不断更新相关文献。



## **15. TAIGen: Training-Free Adversarial Image Generation via Diffusion Models**

TAIGen：通过扩散模型的免训练对抗图像生成 cs.CV

Accepted at ICCVW-CV4BIOM 2025

**SubmitDate**: 2025-08-20    [abs](http://arxiv.org/abs/2508.15020v1) [paper-pdf](http://arxiv.org/pdf/2508.15020v1)

**Authors**: Susim Roy, Anubhooti Jain, Mayank Vatsa, Richa Singh

**Abstract**: Adversarial attacks from generative models often produce low-quality images and require substantial computational resources. Diffusion models, though capable of high-quality generation, typically need hundreds of sampling steps for adversarial generation. This paper introduces TAIGen, a training-free black-box method for efficient adversarial image generation. TAIGen produces adversarial examples using only 3-20 sampling steps from unconditional diffusion models. Our key finding is that perturbations injected during the mixing step interval achieve comparable attack effectiveness without processing all timesteps. We develop a selective RGB channel strategy that applies attention maps to the red channel while using GradCAM-guided perturbations on green and blue channels. This design preserves image structure while maximizing misclassification in target models. TAIGen maintains visual quality with PSNR above 30 dB across all tested datasets. On ImageNet with VGGNet as source, TAIGen achieves 70.6% success against ResNet, 80.8% against MNASNet, and 97.8% against ShuffleNet. The method generates adversarial examples 10x faster than existing diffusion-based attacks. Our method achieves the lowest robust accuracy, indicating it is the most impactful attack as the defense mechanism is least successful in purifying the images generated by TAIGen.

摘要: 来自生成模型的对抗攻击通常会产生低质量的图像，并且需要大量的计算资源。扩散模型虽然能够高质量生成，但通常需要数百个采样步骤来对抗生成。本文介绍了TAIGen，这是一种用于高效生成对抗图像的免训练黑匣子方法。TAIGen仅使用无条件扩散模型的3-20个采样步骤即可生成对抗性示例。我们的关键发现是，在混合步骤间隔期间注入的扰动无需处理所有时间步即可实现相当的攻击有效性。我们开发了一种选择性的RB通道策略，将注意力图应用于红色通道，同时在绿色和蓝色通道上使用GradCAM引导的扰动。该设计保留了图像结构，同时最大化目标模型中的错误分类。TAIGen在所有测试数据集中保持视觉质量，PSNR高于30分贝。在以VGGnet为源的ImageNet上，TAIGen对ResNet的成功率为70.6%，对MNASNet的成功率为80.8%，对ShuffleNet的成功率为97.8%。该方法生成对抗性示例的速度比现有的基于扩散的攻击快10倍。我们的方法实现了最低的鲁棒准确性，这表明它是最有影响力的攻击，因为防御机制在净化TAIGen生成的图像方面最不成功。



## **16. Security Concerns for Large Language Models: A Survey**

大型语言模型的安全问题：调查 cs.CR

**SubmitDate**: 2025-08-20    [abs](http://arxiv.org/abs/2505.18889v4) [paper-pdf](http://arxiv.org/pdf/2505.18889v4)

**Authors**: Miles Q. Li, Benjamin C. M. Fung

**Abstract**: Large Language Models (LLMs) such as ChatGPT and its competitors have caused a revolution in natural language processing, but their capabilities also introduce new security vulnerabilities. This survey provides a comprehensive overview of these emerging concerns, categorizing threats into several key areas: prompt injection and jailbreaking; adversarial attacks, including input perturbations and data poisoning; misuse by malicious actors to generate disinformation, phishing emails, and malware; and the worrisome risks inherent in autonomous LLM agents. Recently, a significant focus is increasingly being placed on the latter, exploring goal misalignment, emergent deception, self-preservation instincts, and the potential for LLMs to develop and pursue covert, misaligned objectives, a behavior known as scheming, which may even persist through safety training. We summarize recent academic and industrial studies from 2022 to 2025 that exemplify each threat, analyze proposed defenses and their limitations, and identify open challenges in securing LLM-based applications. We conclude by emphasizing the importance of advancing robust, multi-layered security strategies to ensure LLMs are safe and beneficial.

摘要: ChatGPT及其竞争对手等大型语言模型（LLM）引发了自然语言处理领域的一场革命，但它们的功能也引入了新的安全漏洞。这项调查全面概述了这些新出现的问题，将威胁分为几个关键领域：即时注入和越狱;对抗性攻击，包括输入干扰和数据中毒;恶意行为者滥用信息、网络钓鱼电子邮件和恶意软件;以及自主LLM代理固有的令人担忧的风险。最近，人们越来越关注后者，探索目标失调、紧急欺骗、自我保护本能，以及LLM制定和追求隐蔽、失调目标的潜力，这种行为被称为阴谋，甚至可以通过安全培训持续存在。我们总结了2022年至2025年期间最近的学术和工业研究，这些研究揭示了每种威胁，分析了拟议的防御措施及其局限性，并确定了保护基于LLM的应用程序方面的公开挑战。最后，我们强调了推进强大的多层安全策略以确保LLM安全且有益的重要性。



## **17. Universal and Transferable Adversarial Attack on Large Language Models Using Exponentiated Gradient Descent**

使用指数梯度下降对大型语言模型进行普遍且可转移的对抗攻击 cs.LG

**SubmitDate**: 2025-08-20    [abs](http://arxiv.org/abs/2508.14853v1) [paper-pdf](http://arxiv.org/pdf/2508.14853v1)

**Authors**: Sajib Biswas, Mao Nishino, Samuel Jacob Chacko, Xiuwen Liu

**Abstract**: As large language models (LLMs) are increasingly deployed in critical applications, ensuring their robustness and safety alignment remains a major challenge. Despite the overall success of alignment techniques such as reinforcement learning from human feedback (RLHF) on typical prompts, LLMs remain vulnerable to jailbreak attacks enabled by crafted adversarial triggers appended to user prompts. Most existing jailbreak methods either rely on inefficient searches over discrete token spaces or direct optimization of continuous embeddings. While continuous embeddings can be given directly to selected open-source models as input, doing so is not feasible for proprietary models. On the other hand, projecting these embeddings back into valid discrete tokens introduces additional complexity and often reduces attack effectiveness. We propose an intrinsic optimization method which directly optimizes relaxed one-hot encodings of the adversarial suffix tokens using exponentiated gradient descent coupled with Bregman projection, ensuring that the optimized one-hot encoding of each token always remains within the probability simplex. We provide theoretical proof of convergence for our proposed method and implement an efficient algorithm that effectively jailbreaks several widely used LLMs. Our method achieves higher success rates and faster convergence compared to three state-of-the-art baselines, evaluated on five open-source LLMs and four adversarial behavior datasets curated for evaluating jailbreak methods. In addition to individual prompt attacks, we also generate universal adversarial suffixes effective across multiple prompts and demonstrate transferability of optimized suffixes to different LLMs.

摘要: 随着大型语言模型（LLM）越来越多地部署在关键应用程序中，确保其稳健性和安全性一致性仍然是一个重大挑战。尽管典型提示上的人类反馈强化学习（RL HF）等对齐技术取得了总体成功，但LLM仍然容易受到附加在用户提示上的精心设计的对抗触发器所实现的越狱攻击。大多数现有的越狱方法要么依赖于对离散令牌空间的低效搜索，要么依赖于连续嵌入的直接优化。虽然连续嵌入可以直接提供给选定的开源模型作为输入，但这样做对于专有模型来说是不可行的。另一方面，将这些嵌入投影回有效的离散令牌会带来额外的复杂性，并且通常会降低攻击有效性。我们提出了一种内在优化方法，该方法使用指数梯度下降结合布雷格曼投影直接优化对抗性后缀令牌的宽松一次性编码，确保每个令牌的优化一次性编码始终保持在概率单形内。我们为我们提出的方法提供了收敛性的理论证明，并实现了一种有效的算法，可以有效地越狱几种广泛使用的LLM。与三个最先进的基线相比，我们的方法实现了更高的成功率和更快的收敛，这些基线在五个开源LLM和为评估越狱方法而策划的四个对抗行为数据集上进行了评估。除了单独的提示攻击外，我们还生成在多个提示中有效的通用对抗性后缀，并演示优化后缀到不同LLM的可移植性。



## **18. Fragile, Robust, and Antifragile: A Perspective from Parameter Responses in Reinforcement Learning Under Stress**

脆弱性、稳健性和反脆弱性：从压力下强化学习中的参数响应的角度来看 cs.LG

Withdrawn pending a review of attribution and overlap with Pravin et  al., Artificial Intelligence (2024), DOI: 10.1016/j.artint.2023.104060.  Further dissemination is paused while we determine appropriate next steps

**SubmitDate**: 2025-08-20    [abs](http://arxiv.org/abs/2506.23036v2) [paper-pdf](http://arxiv.org/pdf/2506.23036v2)

**Authors**: Zain ul Abdeen, Ming Jin

**Abstract**: This paper explores Reinforcement learning (RL) policy robustness by systematically analyzing network parameters under internal and external stresses. Inspired by synaptic plasticity in neuroscience, synaptic filtering introduces internal stress by selectively perturbing parameters, while adversarial attacks apply external stress through modified agent observations. This dual approach enables the classification of parameters as fragile, robust, or antifragile, based on their influence on policy performance in clean and adversarial settings. Parameter scores are defined to quantify these characteristics, and the framework is validated on PPO-trained agents in Mujoco continuous control environments. The results highlight the presence of antifragile parameters that enhance policy performance under stress, demonstrating the potential of targeted filtering techniques to improve RL policy adaptability. These insights provide a foundation for future advancements in the design of robust and antifragile RL systems.

摘要: 本文通过系统分析内部和外部压力下的网络参数来探索强化学习（RL）策略的稳健性。受神经科学中突触可塑性的启发，突触过滤通过选择性地扰乱参数来引入内部压力，而对抗攻击则通过修改的主体观察来施加外部压力。这种双重方法可以根据参数在干净和敌对环境中对政策绩效的影响将参数分类为脆弱、稳健或反脆弱。定义参数分数来量化这些特征，并在Mujoco连续控制环境中的PPA训练代理上验证了该框架。结果强调了反脆弱参数的存在，可以增强压力下的政策性能，证明了有针对性的过滤技术提高RL政策适应性的潜力。这些见解为未来鲁棒和抗脆弱RL系统设计的进步提供了基础。



## **19. Distributional Adversarial Attacks and Training in Deep Hedging**

分布式对抗攻击和深度对冲培训 math.OC

Preprint. Under review

**SubmitDate**: 2025-08-20    [abs](http://arxiv.org/abs/2508.14757v1) [paper-pdf](http://arxiv.org/pdf/2508.14757v1)

**Authors**: Guangyi He, Tobias Sutter, Lukas Gonon

**Abstract**: In this paper, we study the robustness of classical deep hedging strategies under distributional shifts by leveraging the concept of adversarial attacks. We first demonstrate that standard deep hedging models are highly vulnerable to small perturbations in the input distribution, resulting in significant performance degradation. Motivated by this, we propose an adversarial training framework tailored to increase the robustness of deep hedging strategies. Our approach extends pointwise adversarial attacks to the distributional setting and introduces a computationally tractable reformulation of the adversarial optimization problem over a Wasserstein ball. This enables the efficient training of hedging strategies that are resilient to distributional perturbations. Through extensive numerical experiments, we show that adversarially trained deep hedging strategies consistently outperform their classical counterparts in terms of out-of-sample performance and resilience to model misspecification. Our findings establish a practical and effective framework for robust deep hedging under realistic market uncertainties.

摘要: 在本文中，我们利用对抗性攻击的概念研究了经典深度对冲策略在分布变化下的鲁棒性。我们首先证明了标准的深度对冲模型非常容易受到输入分布中的小扰动的影响，从而导致显着的性能下降。受此启发，我们提出了一个对抗性训练框架，以提高深度对冲策略的鲁棒性。我们的方法将逐点对抗攻击扩展到分布式设置，并在Wasserstein球上引入了对抗优化问题的计算上易于处理的重新表述。这使得能够有效地训练对分布扰动有弹性的对冲策略。通过大量的数值实验，我们表明，在样本外性能和对模型错误指定的弹性方面，对抗训练的深度对冲策略始终优于经典对冲策略。我们的研究结果建立了一个实用和有效的框架下，现实的市场不确定性的强大的深度对冲。



## **20. Foe for Fraud: Transferable Adversarial Attacks in Credit Card Fraud Detection**

欺诈的敌人：信用卡欺诈检测中的可转移对抗攻击 cs.CR

**SubmitDate**: 2025-08-20    [abs](http://arxiv.org/abs/2508.14699v1) [paper-pdf](http://arxiv.org/pdf/2508.14699v1)

**Authors**: Jan Lum Fok, Qingwen Zeng, Shiping Chen, Oscar Fawkes, Huaming Chen

**Abstract**: Credit card fraud detection (CCFD) is a critical application of Machine Learning (ML) in the financial sector, where accurately identifying fraudulent transactions is essential for mitigating financial losses. ML models have demonstrated their effectiveness in fraud detection task, in particular with the tabular dataset. While adversarial attacks have been extensively studied in computer vision and deep learning, their impacts on the ML models, particularly those trained on CCFD tabular datasets, remains largely unexplored. These latent vulnerabilities pose significant threats to the security and stability of the financial industry, especially in high-value transactions where losses could be substantial. To address this gap, in this paper, we present a holistic framework that investigate the robustness of CCFD ML model against adversarial perturbations under different circumstances. Specifically, the gradient-based attack methods are incorporated into the tabular credit card transaction data in both black- and white-box adversarial attacks settings. Our findings confirm that tabular data is also susceptible to subtle perturbations, highlighting the need for heightened awareness among financial technology practitioners regarding ML model security and trustworthiness. Furthermore, the experiments by transferring adversarial samples from gradient-based attack method to non-gradient-based models also verify our findings. Our results demonstrate that such attacks remain effective, emphasizing the necessity of developing robust defenses for CCFD algorithms.

摘要: 信用卡欺诈检测（CCFD）是机器学习（ML）在金融领域的重要应用，准确识别欺诈交易对于减轻财务损失至关重要。ML模型已经证明了它们在欺诈检测任务中的有效性，特别是对于表格数据集。虽然对抗性攻击在计算机视觉和深度学习中得到了广泛的研究，但它们对ML模型的影响，尤其是在CCFD表格数据集上训练的ML模型，在很大程度上仍未被探索。这些潜在漏洞对金融行业的安全和稳定构成重大威胁，尤其是在损失可能巨大的高价值交易中。为了解决这一差距，在本文中，我们提出了一个整体框架，研究CCFD ML模型在不同情况下对抗性扰动的稳健性。具体来说，在黑箱和白箱对抗攻击设置中，基于梯度的攻击方法都被合并到表格式信用卡交易数据中。我们的研究结果证实，表格数据也容易受到微妙的干扰，这凸显了金融技术从业者需要提高对ML模型安全性和可信性的认识。此外，将对抗样本从基于梯度的攻击方法转移到非基于梯度的模型的实验也验证了我们的发现。我们的结果表明此类攻击仍然有效，强调了为CCFD算法开发稳健防御的必要性。



## **21. Dark Miner: Defend against undesirable generation for text-to-image diffusion models**

Dark Miner：防止文本到图像扩散模型的不良生成 cs.CV

**SubmitDate**: 2025-08-20    [abs](http://arxiv.org/abs/2409.17682v3) [paper-pdf](http://arxiv.org/pdf/2409.17682v3)

**Authors**: Zheling Meng, Bo Peng, Xiaochuan Jin, Yue Jiang, Wei Wang, Jing Dong, Tieniu Tan

**Abstract**: Text-to-image diffusion models have been demonstrated with undesired generation due to unfiltered large-scale training data, such as sexual images and copyrights, necessitating the erasure of undesired concepts. Most existing methods focus on modifying the generation probabilities conditioned on the texts containing target concepts. However, they fail to guarantee the desired generation of texts unseen in the training phase, especially for the adversarial texts from malicious attacks. In this paper, we analyze the erasure task and point out that existing methods cannot guarantee the minimization of the total probabilities of undesired generation. To tackle this problem, we propose Dark Miner. It entails a recurring three-stage process that comprises mining, verifying, and circumventing. This method greedily mines embeddings with maximum generation probabilities of target concepts and more effectively reduces their generation. In the experiments, we evaluate its performance on the inappropriateness, object, and style concepts. Compared with the previous methods, our method achieves better erasure and defense results, especially under multiple adversarial attacks, while preserving the native generation capability of the models. Our code will be available on GitHub.

摘要: 由于未经过滤的大规模训练数据（例如性图像和版权），文本到图像扩散模型已经被证明会产生不希望的生成，这使得必须擦除不希望的概念。大多数现有的方法都集中在修改以包含目标概念的文本为条件的生成概率上。然而，它们无法保证在训练阶段生成所需的不可见的文本，尤其是对于来自恶意攻击的对抗文本。本文分析了擦除任务，指出现有方法无法保证不期望生成的总概率最小化。为了解决这个问题，我们提出了Dark Miner。它需要一个重复的三阶段过程，包括挖掘、验证和规避。该方法能更好地挖掘目标概念生成概率最大的嵌入，更有效地减少其生成。在实验中，我们评估了它的表现上的不适当性，对象和风格的概念。与以前的方法相比，我们的方法取得了更好的擦除和防御的结果，特别是在多个对抗性攻击，同时保留了原生的生成能力的模型。我们的代码将在GitHub上提供。



## **22. When Good Sounds Go Adversarial: Jailbreaking Audio-Language Models with Benign Inputs**

当好的声音变得敌对时：用良性输入越狱的音频模型 cs.SD

**SubmitDate**: 2025-08-20    [abs](http://arxiv.org/abs/2508.03365v2) [paper-pdf](http://arxiv.org/pdf/2508.03365v2)

**Authors**: Bodam Kim, Hiskias Dingeto, Taeyoun Kwon, Dasol Choi, DongGeon Lee, Haon Park, JaeHoon Lee, Jongho Shin

**Abstract**: As large language models become increasingly integrated into daily life, audio has emerged as a key interface for human-AI interaction. However, this convenience also introduces new vulnerabilities, making audio a potential attack surface for adversaries. Our research introduces WhisperInject, a two-stage adversarial audio attack framework that can manipulate state-of-the-art audio language models to generate harmful content. Our method uses imperceptible perturbations in audio inputs that remain benign to human listeners. The first stage uses a novel reward-based optimization method, Reinforcement Learning with Projected Gradient Descent (RL-PGD), to guide the target model to circumvent its own safety protocols and generate harmful native responses. This native harmful response then serves as the target for Stage 2, Payload Injection, where we use Projected Gradient Descent (PGD) to optimize subtle perturbations that are embedded into benign audio carriers, such as weather queries or greeting messages. Validated under the rigorous StrongREJECT, LlamaGuard, as well as Human Evaluation safety evaluation framework, our experiments demonstrate a success rate exceeding 86% across Qwen2.5-Omni-3B, Qwen2.5-Omni-7B, and Phi-4-Multimodal. Our work demonstrates a new class of practical, audio-native threats, moving beyond theoretical exploits to reveal a feasible and covert method for manipulating AI behavior.

摘要: 随着大型语言模型越来越融入日常生活，音频已成为人机交互的关键界面。然而，这种便利性也引入了新的漏洞，使音频成为对手的潜在攻击面。我们的研究引入了WhisperInib，这是一个两阶段对抗性音频攻击框架，可以操纵最先进的音频语言模型来生成有害内容。我们的方法在音频输入中使用不可感知的扰动，这些扰动对人类听众保持良性。第一阶段使用一种新颖的基于奖励的优化方法--具有投影梯度下降的强化学习（RL-PVD），来指导目标模型规避其自己的安全协议并生成有害的原生响应。然后，这种原生有害响应作为第二阶段有效负载注入的目标，在该阶段，我们使用投影梯度下降（PVD）来优化嵌入良性音频载体中的微妙扰动，例如天气查询或问候消息。我们的实验经过严格的StrongRESEARCH、LlamaGuard以及Human Evision安全评估框架的验证，证明Qwen 2.5-Omni-3B、Qwen 2.5-Omni-7 B和Phi-4-Multimodal的成功率超过86%。我们的工作展示了一类新的实用、音频原生威胁，超越了理论利用，揭示了一种可行且隐蔽的操纵人工智能行为的方法。



## **23. Adversarial control of synchronization in complex oscillator networks**

复振子网络同步的对抗控制 nlin.AO

10 pages, 4 figures

**SubmitDate**: 2025-08-20    [abs](http://arxiv.org/abs/2506.02403v2) [paper-pdf](http://arxiv.org/pdf/2506.02403v2)

**Authors**: Yasutoshi Nagahama, Kosuke Miyazato, Kazuhiro Takemoto

**Abstract**: This study investigates adversarial attacks, a concept from deep learning, designed to control synchronization dynamics through strategically crafted weak perturbations. We propose a gradient-based optimization method that identifies small phase perturbations to dramatically enhance or suppress collective synchronization in Kuramoto oscillator networks. Our approach formulates synchronization control as an adversarial optimization problem, computing gradients of the order parameter with respect to oscillator phases to determine optimal perturbation directions. Results demonstrate that extremely small phase perturbations applied to network oscillators can achieve significant synchronization control across diverse network architectures. Our analysis reveals that synchronization enhancement is achievable across various network sizes, while synchronization suppression becomes particularly effective in larger networks, with effectiveness scaling favorably with network size. The method is systematically validated on canonical model networks including scale-free and small-world topologies, and real-world networks representing power grids and brain connectivity patterns. This adversarial framework represents a novel paradigm for synchronization management by introducing deep learning concepts to networked dynamical systems.

摘要: 这项研究调查了对抗性攻击，这是一个来自深度学习的概念，旨在通过策略性精心设计的弱扰动来控制同步动态。我们提出了一种基于梯度的优化方法，该方法识别小的相扰动，以显着增强或抑制仓本振荡器网络中的集体同步。我们的方法将同步控制制定为一个对抗优化问题，计算阶参数相对于振荡器相的梯度以确定最佳扰动方向。结果表明，应用于网络振荡器的极小的相扰动可以实现跨不同网络架构的显着同步控制。我们的分析表明，同步增强可以在不同的网络规模中实现，而同步抑制在更大的网络中变得特别有效，并且有效性随着网络规模的增加而有利地扩展。该方法在规范模型网络（包括无标度和小世界布局）以及代表电网和大脑连接模式的现实世界网络上进行了系统验证。这种对抗性框架通过将深度学习概念引入网络动态系统，代表了一种新颖的同步管理范式。



## **24. Backdooring Self-Supervised Contrastive Learning by Noisy Alignment**

通过噪音对齐进行后门自我监督对比学习 cs.CV

Accepted by ICCV 2025

**SubmitDate**: 2025-08-19    [abs](http://arxiv.org/abs/2508.14015v1) [paper-pdf](http://arxiv.org/pdf/2508.14015v1)

**Authors**: Tuo Chen, Jie Gui, Minjing Dong, Ju Jia, Lanting Fang, Jian Liu

**Abstract**: Self-supervised contrastive learning (CL) effectively learns transferable representations from unlabeled data containing images or image-text pairs but suffers vulnerability to data poisoning backdoor attacks (DPCLs). An adversary can inject poisoned images into pretraining datasets, causing compromised CL encoders to exhibit targeted misbehavior in downstream tasks. Existing DPCLs, however, achieve limited efficacy due to their dependence on fragile implicit co-occurrence between backdoor and target object and inadequate suppression of discriminative features in backdoored images. We propose Noisy Alignment (NA), a DPCL method that explicitly suppresses noise components in poisoned images. Inspired by powerful training-controllable CL attacks, we identify and extract the critical objective of noisy alignment, adapting it effectively into data-poisoning scenarios. Our method implements noisy alignment by strategically manipulating contrastive learning's random cropping mechanism, formulating this process as an image layout optimization problem with theoretically derived optimal parameters. The resulting method is simple yet effective, achieving state-of-the-art performance compared to existing DPCLs, while maintaining clean-data accuracy. Furthermore, Noisy Alignment demonstrates robustness against common backdoor defenses. Codes can be found at https://github.com/jsrdcht/Noisy-Alignment.

摘要: 自监督对比学习（CL）有效地从包含图像或图像-文本对的未标记数据中学习可转移表示，但容易受到数据中毒后门攻击（EDL）的攻击。对手可以将有毒图像注入预训练数据集中，导致受损的CL编码器在下游任务中表现出有针对性的不当行为。然而，现有的TLR由于依赖于后门和目标对象之间脆弱的隐式共生以及后门图像中的区分特征抑制不足，其功效有限。我们提出了噪音对齐（NA），这是一种DPDL方法，可以显式地抑制中毒图像中的噪音成分。受到强大的训练可控CL攻击的启发，我们识别并提取有噪对齐的关键目标，将其有效地适应数据中毒场景。我们的方法通过策略性地操纵对比学习的随机裁剪机制来实现有噪的对齐，将这个过程描述为具有理论上推导出的最佳参数的图像布局优化问题。由此产生的方法简单而有效，与现有的TLR相比，实现了最先进的性能，同时保持了干净的数据准确性。此外，Noisy Alliance展示了针对常见后门防御的鲁棒性。代码可在https://github.com/jsrdcht/Noisy-Alignment上找到。



## **25. FedUP: Efficient Pruning-based Federated Unlearning for Model Poisoning Attacks**

FedUP：针对模型中毒攻击的高效基于修剪的联邦撤销学习 cs.LG

15 pages, 5 figures, 7 tables

**SubmitDate**: 2025-08-19    [abs](http://arxiv.org/abs/2508.13853v1) [paper-pdf](http://arxiv.org/pdf/2508.13853v1)

**Authors**: Nicolò Romandini, Cristian Borcea, Rebecca Montanari, Luca Foschini

**Abstract**: Federated Learning (FL) can be vulnerable to attacks, such as model poisoning, where adversaries send malicious local weights to compromise the global model. Federated Unlearning (FU) is emerging as a solution to address such vulnerabilities by selectively removing the influence of detected malicious contributors on the global model without complete retraining. However, unlike typical FU scenarios where clients are trusted and cooperative, applying FU with malicious and possibly colluding clients is challenging because their collaboration in unlearning their data cannot be assumed. This work presents FedUP, a lightweight FU algorithm designed to efficiently mitigate malicious clients' influence by pruning specific connections within the attacked model. Our approach achieves efficiency by relying only on clients' weights from the last training round before unlearning to identify which connections to inhibit. Isolating malicious influence is non-trivial due to overlapping updates from benign and malicious clients. FedUP addresses this by carefully selecting and zeroing the highest magnitude weights that diverge the most between the latest updates from benign and malicious clients while preserving benign information. FedUP is evaluated under a strong adversarial threat model, where up to 50%-1 of the clients could be malicious and have full knowledge of the aggregation process. We demonstrate the effectiveness, robustness, and efficiency of our solution through experiments across IID and Non-IID data, under label-flipping and backdoor attacks, and by comparing it with state-of-the-art (SOTA) FU solutions. In all scenarios, FedUP reduces malicious influence, lowering accuracy on malicious data to match that of a model retrained from scratch while preserving performance on benign data. FedUP achieves effective unlearning while consistently being faster and saving storage compared to the SOTA.

摘要: 联邦学习（FL）可能容易受到攻击，例如模型中毒，其中对手发送恶意本地权重来损害全局模型。联合取消学习（FU）正在成为一种解决此类漏洞的解决方案，通过选择性地消除检测到的恶意贡献者对全球模型的影响，而无需完全重新培训。然而，与客户值得信任和合作的典型FU场景不同，对恶意且可能勾结的客户应用FU具有挑战性，因为无法假设他们在忘记数据方面进行了合作。这项工作提出了FedUP，这是一种轻量级FU算法，旨在通过修剪受攻击模型中的特定连接来有效减轻恶意客户端的影响。我们的方法通过仅依赖于客户端在最后一轮训练中的权重来实现效率，然后再进行学习以确定要抑制哪些连接。由于良性和恶意客户端的更新重叠，因此隔离恶意影响并不简单。FedUP通过仔细选择和归零来自良性和恶意客户端的最新更新之间差异最大的最高幅度权重来解决这个问题，同时保留良性信息。FedUP在强对抗性威胁模型下进行评估，其中高达50%-1的客户端可能是恶意的，并且完全了解聚合过程。我们通过在标签翻转和后门攻击下跨IID和非IID数据进行实验，并将其与最先进的（SOTA）FU解决方案进行比较，证明了我们解决方案的有效性、稳健性和效率。在所有场景中，FedUP都会减少恶意影响，降低恶意数据的准确性，以匹配从头重新训练的模型的准确性，同时保留良性数据的性能。与SOTA相比，FedUP实现了有效的取消学习，同时始终更快并节省存储。



## **26. Timestep-Compressed Attack on Spiking Neural Networks through Timestep-Level Backpropagation**

通过时步级反向传播对尖峰神经网络的时步压缩攻击 cs.CV

8 pages

**SubmitDate**: 2025-08-19    [abs](http://arxiv.org/abs/2508.13812v1) [paper-pdf](http://arxiv.org/pdf/2508.13812v1)

**Authors**: Donghwa Kang, Doohyun Kim, Sang-Ki Ko, Jinkyu Lee, Hyeongboo Baek, Brent ByungHoon Kang

**Abstract**: State-of-the-art (SOTA) gradient-based adversarial attacks on spiking neural networks (SNNs), which largely rely on extending FGSM and PGD frameworks, face a critical limitation: substantial attack latency from multi-timestep processing, rendering them infeasible for practical real-time applications. This inefficiency stems from their design as direct extensions of ANN paradigms, which fail to exploit key SNN properties. In this paper, we propose the timestep-compressed attack (TCA), a novel framework that significantly reduces attack latency. TCA introduces two components founded on key insights into SNN behavior. First, timestep-level backpropagation (TLBP) is based on our finding that global temporal information in backpropagation to generate perturbations is not critical for an attack's success, enabling per-timestep evaluation for early stopping. Second, adversarial membrane potential reuse (A-MPR) is motivated by the observation that initial timesteps are inefficiently spent accumulating membrane potential, a warm-up phase that can be pre-calculated and reused. Our experiments on VGG-11 and ResNet-17 with the CIFAR-10/100 and CIFAR10-DVS datasets show that TCA significantly reduces the required attack latency by up to 56.6% and 57.1% compared to SOTA methods in white-box and black-box settings, respectively, while maintaining a comparable attack success rate.

摘要: 对尖峰神经网络（SNN）的最新技术（SOTA）基于梯度的对抗攻击主要依赖于扩展FGSM和PVD框架，但面临着一个严重的限制：来自多时步处理的大量攻击延迟，使其不适用于实际的实时应用程序。这种低效率源于它们的设计作为NN范式的直接扩展，而该范式未能利用关键的SNN属性。在本文中，我们提出了时步压缩攻击（MCA），这是一种显着减少攻击延迟的新型框架。MCA引入了两个基于对SNN行为的关键见解的组件。首先，时步级反向传播（TLBP）基于我们的发现，即反向传播中生成扰动的全局时间信息对于攻击的成功并不关键，从而能够进行按时步评估以提前停止。其次，对抗性膜势再利用（A-MPI）的动机是这样一种观察：初始时间步被低效地用于积累膜势，这是一个可以预先计算和重复使用的预热阶段。我们使用CIFAR-10/100和CIFAR 10-DVS数据集对VGG-11和ResNet-17进行的实验表明，与白盒和黑盒设置中的SOTA方法相比，MCA分别将所需的攻击延迟显着降低了56.6%和57.1%，同时保持了相当的攻击成功率。



## **27. Enhancing Targeted Adversarial Attacks on Large Vision-Language Models through Intermediate Projector Guidance**

通过中间投影仪指导增强对大型视觉语言模型的有针对性的对抗攻击 cs.CV

**SubmitDate**: 2025-08-19    [abs](http://arxiv.org/abs/2508.13739v1) [paper-pdf](http://arxiv.org/pdf/2508.13739v1)

**Authors**: Yiming Cao, Yanjie Li, Kaisheng Liang, Yuni Lai, Bin Xiao

**Abstract**: Targeted adversarial attacks are essential for proactively identifying security flaws in Vision-Language Models before real-world deployment. However, current methods perturb images to maximize global similarity with the target text or reference image at the encoder level, collapsing rich visual semantics into a single global vector. This limits attack granularity, hindering fine-grained manipulations such as modifying a car while preserving its background. Furthermore, these methods largely overlook the projector module, a critical semantic bridge between the visual encoder and the language model in VLMs, thereby failing to disrupt the full vision-language alignment pipeline within VLMs and limiting attack effectiveness. To address these issues, we propose the Intermediate Projector Guided Attack (IPGA), the first method to attack using the intermediate stage of the projector module, specifically the widely adopted Q-Former, which transforms global image embeddings into fine-grained visual features. This enables more precise control over adversarial perturbations by operating on semantically meaningful visual tokens rather than a single global representation. Specifically, IPGA leverages the Q-Former pretrained solely on the first vision-language alignment stage, without LLM fine-tuning, which improves both attack effectiveness and transferability across diverse VLMs. Furthermore, we propose Residual Query Alignment (RQA) to preserve unrelated visual content, thereby yielding more controlled and precise adversarial manipulations. Extensive experiments show that our attack method consistently outperforms existing methods in both standard global image captioning tasks and fine-grained visual question-answering tasks in black-box environment. Additionally, IPGA successfully transfers to multiple commercial VLMs, including Google Gemini and OpenAI GPT.

摘要: 有针对性的对抗攻击对于在现实世界部署之前主动识别视觉语言模型中的安全缺陷至关重要。然而，当前的方法会扰乱图像，以在编码器级别最大化与目标文本或参考图像的全局相似性，将丰富的视觉语义折叠到单个全局载体中。这限制了攻击粒度，阻碍了细粒度操作，例如在保留背景的同时修改汽车。此外，这些方法在很大程度上忽视了投影仪模块，这是VLM中视觉编码器和语言模型之间的关键语义桥梁，从而无法破坏VLM内的完整视觉-语言对齐管道并限制攻击有效性。为了解决这些问题，我们提出了中间投影仪引导攻击（IPGA），这是第一种使用投影仪模块中间阶段进行攻击的方法，特别是广泛采用的Q-Former，它将全局图像嵌入转换为细粒度视觉特征。这使得通过对具有语义意义的视觉标记而不是单个全局表示进行操作，能够更精确地控制对抗性扰动。具体来说，IPGA利用仅在第一个视觉语言对齐阶段预训练的Q-Former，无需LLM微调，从而提高了攻击有效性和跨不同VLM的可移植性。此外，我们提出了剩余查询对齐（RQA）来保留不相关的视觉内容，从而产生更受控和更精确的对抗性操纵。大量实验表明，我们的攻击方法在标准全局图像字幕任务和黑匣子环境中的细粒度视觉问答任务中始终优于现有方法。此外，IPGA还成功转移到多个商业VLM，包括Google Gemini和OpenAI GPT。



## **28. The AI Risk Spectrum: From Dangerous Capabilities to Existential Threats**

人工智能风险谱：从危险能力到潜在威胁 cs.CY

**SubmitDate**: 2025-08-19    [abs](http://arxiv.org/abs/2508.13700v1) [paper-pdf](http://arxiv.org/pdf/2508.13700v1)

**Authors**: Markov Grey, Charbel-Raphaël Segerie

**Abstract**: As AI systems become more capable, integrated, and widespread, understanding the associated risks becomes increasingly important. This paper maps the full spectrum of AI risks, from current harms affecting individual users to existential threats that could endanger humanity's survival. We organize these risks into three main causal categories. Misuse risks, which occur when people deliberately use AI for harmful purposes - creating bioweapons, launching cyberattacks, adversarial AI attacks or deploying lethal autonomous weapons. Misalignment risks happen when AI systems pursue outcomes that conflict with human values, irrespective of developer intentions. This includes risks arising through specification gaming (reward hacking), scheming and power-seeking tendencies in pursuit of long-term strategic goals. Systemic risks, which arise when AI integrates into complex social systems in ways that gradually undermine human agency - concentrating power, accelerating political and economic disempowerment, creating overdependence that leads to human enfeeblement, or irreversibly locking in current values curtailing future moral progress. Beyond these core categories, we identify risk amplifiers - competitive pressures, accidents, corporate indifference, and coordination failures - that make all risks more likely and severe. Throughout, we connect today's existing risks and empirically observable AI behaviors to plausible future outcomes, demonstrating how existing trends could escalate to catastrophic outcomes. Our goal is to help readers understand the complete landscape of AI risks. Good futures are possible, but they don't happen by default. Navigating these challenges will require unprecedented coordination, but an extraordinary future awaits if we do.

摘要: 随着人工智能系统变得越来越强大、集成化和广泛，了解相关风险变得越来越重要。本文描绘了人工智能风险的全方位，从当前影响个人用户的伤害到可能危及人类生存的生存威胁。我们将这些风险分为三个主要因果类别。滥用风险，当人们故意将人工智能用于有害目的时，就会出现这种风险--制造生物武器、发动网络攻击、对抗性人工智能攻击或部署致命的自主武器。当人工智能系统追求与人类价值观相冲突的结果时，无论开发人员的意图如何，就会发生错位风险。这包括为追求长期战略目标而制定的规范游戏（奖励黑客攻击）、阴谋和权力追求倾向所产生的风险。系统性风险，当人工智能以逐渐削弱人类代理力的方式融入复杂的社会系统时，就会出现系统性风险--集中权力、加速政治和经济权力丧失、造成过度依赖导致人类衰弱，或者不可逆转地锁定当前价值观，从而限制未来的道德进步。除了这些核心类别之外，我们还确定了风险放大器--竞争压力、事故、企业冷漠和协调失败--这些因素使所有风险变得更有可能和更严重。自始至终，我们将当今现有的风险和经验上可观察的人工智能行为与合理的未来结果联系起来，展示了现有趋势如何升级为灾难性结果。我们的目标是帮助读者了解人工智能风险的完整格局。美好的未来是可能的，但它们不会默认发生。应对这些挑战需要前所未有的协调，但如果我们这样做，一个非凡的未来就在等待着我们。



## **29. Mask and Restore: Blind Backdoor Defense at Test Time with Masked Autoencoder**

屏蔽和恢复：使用屏蔽自动编码器在测试时进行盲后门防御 cs.LG

**SubmitDate**: 2025-08-19    [abs](http://arxiv.org/abs/2303.15564v3) [paper-pdf](http://arxiv.org/pdf/2303.15564v3)

**Authors**: Tao Sun, Lu Pang, Weimin Lyu, Chao Chen, Haibin Ling

**Abstract**: Deep neural networks are vulnerable to backdoor attacks, where an adversary manipulates the model behavior through overlaying images with special triggers. Existing backdoor defense methods often require accessing a few validation data and model parameters, which is impractical in many real-world applications, e.g., when the model is provided as a cloud service. In this paper, we address the practical task of blind backdoor defense at test time, in particular for local attacks and black-box models. The true label of every test image needs to be recovered on the fly from a suspicious model regardless of image benignity. We consider test-time image purification that incapacitates local triggers while keeping semantic contents intact. Due to diverse trigger patterns and sizes, the heuristic trigger search can be unscalable. We circumvent such barrier by leveraging the strong reconstruction power of generative models, and propose Blind Defense with Masked AutoEncoder (BDMAE). BDMAE detects possible local triggers using image structural similarity and label consistency between the test image and MAE restorations. The detection results are then refined by considering trigger topology. Finally, we fuse MAE restorations adaptively into a purified image for making prediction. Extensive experiments under different backdoor settings validate its effectiveness and generalizability.

摘要: 深度神经网络容易受到后门攻击，其中对手通过使用特殊触发器覆盖图像来操纵模型行为。现有的后门防御方法通常需要访问一些验证数据和模型参数，这在许多现实世界的应用中是不切实际的，例如，当模型作为云服务提供时。在本文中，我们解决了实际任务的盲后门防御在测试时，特别是本地攻击和黑盒模型。每张测试图像的真实标签都需要从可疑模型中立即恢复，无论图像是否友善。我们考虑测试时图像净化，使本地触发器失去能力，同时保持语义内容完整。由于触发模式和大小不同，启发式触发搜索可能无法扩展。我们通过利用生成式模型的强大重建能力来规避这一障碍，并提出使用Masked AutoEncoder（BCMAE）的盲防御。BCMAE使用测试图像和MAE解释之间的图像结构相似性和标签一致性来检测可能的局部触发。然后通过考虑触发器布局来细化检测结果。最后，我们将MAE分解自适应地融合到净化图像中进行预测。不同后门设置下的大量实验验证了其有效性和可推广性。



## **30. Robust Federated Learning under Adversarial Attacks via Loss-Based Client Clustering**

通过基于损失的客户端集群实现对抗性攻击下的鲁棒联邦学习 cs.LG

16 pages, 5 figures

**SubmitDate**: 2025-08-19    [abs](http://arxiv.org/abs/2508.12672v2) [paper-pdf](http://arxiv.org/pdf/2508.12672v2)

**Authors**: Emmanouil Kritharakis, Dusan Jakovetic, Antonios Makris, Konstantinos Tserpes

**Abstract**: Federated Learning (FL) enables collaborative model training across multiple clients without sharing private data. We consider FL scenarios wherein FL clients are subject to adversarial (Byzantine) attacks, while the FL server is trusted (honest) and has a trustworthy side dataset. This may correspond to, e.g., cases where the server possesses trusted data prior to federation, or to the presence of a trusted client that temporarily assumes the server role. Our approach requires only two honest participants, i.e., the server and one client, to function effectively, without prior knowledge of the number of malicious clients. Theoretical analysis demonstrates bounded optimality gaps even under strong Byzantine attacks. Experimental results show that our algorithm significantly outperforms standard and robust FL baselines such as Mean, Trimmed Mean, Median, Krum, and Multi-Krum under various attack strategies including label flipping, sign flipping, and Gaussian noise addition across MNIST, FMNIST, and CIFAR-10 benchmarks using the Flower framework.

摘要: 联合学习（FL）支持跨多个客户端的协作模型训练，而无需共享私有数据。我们考虑FL场景，其中FL客户端受到对抗性（拜占庭）攻击，而FL服务器是可信的（诚实的），并有一个值得信赖的侧数据集。这可以对应于，例如，服务器在联合之前拥有受信任数据，或者存在暂时承担服务器角色的受信任客户端的情况。我们的方法只需要两个诚实的参与者，即服务器和一个客户端，在不了解恶意客户端数量的情况下有效运行。理论分析表明，即使在强大的拜占庭攻击下，也存在有限的最优性差距。实验结果表明，我们的算法显着优于标准和强大的FL基线，如平均值，修剪平均值，中位数，克鲁姆，和多克鲁姆下的各种攻击策略，包括标签翻转，符号翻转，高斯噪声添加在MNIST，FMNIST，和CIFAR-10基准使用花框架。



## **31. CCFC: Core & Core-Full-Core Dual-Track Defense for LLM Jailbreak Protection**

CCFC：核心与核心-全核心双轨防御LLM越狱保护 cs.CR

11 pages, 1 figure

**SubmitDate**: 2025-08-19    [abs](http://arxiv.org/abs/2508.14128v1) [paper-pdf](http://arxiv.org/pdf/2508.14128v1)

**Authors**: Jiaming Hu, Haoyu Wang, Debarghya Mukherjee, Ioannis Ch. Paschalidis

**Abstract**: Jailbreak attacks pose a serious challenge to the safe deployment of large language models (LLMs). We introduce CCFC (Core & Core-Full-Core), a dual-track, prompt-level defense framework designed to mitigate LLMs' vulnerabilities from prompt injection and structure-aware jailbreak attacks. CCFC operates by first isolating the semantic core of a user query via few-shot prompting, and then evaluating the query using two complementary tracks: a core-only track to ignore adversarial distractions (e.g., toxic suffixes or prefix injections), and a core-full-core (CFC) track to disrupt the structural patterns exploited by gradient-based or edit-based attacks. The final response is selected based on a safety consistency check across both tracks, ensuring robustness without compromising on response quality. We demonstrate that CCFC cuts attack success rates by 50-75% versus state-of-the-art defenses against strong adversaries (e.g., DeepInception, GCG), without sacrificing fidelity on benign queries. Our method consistently outperforms state-of-the-art prompt-level defenses, offering a practical and effective solution for safer LLM deployment.

摘要: 越狱攻击对大型语言模型（LLM）的安全部署构成了严重挑战。我们引入了CCFC（Core & Core-Full-Core），这是一种双轨预算级防御框架，旨在缓解LLM免受即时注入和结构感知越狱攻击的漏洞。CCFC的运作方式是首先通过少量提示隔离用户查询的语义核心，然后使用两个补充的轨道来评估查询：仅核心轨道以忽略对抗干扰（例如，有毒后缀或前置注入），以及核心-全核心（CFC）轨道，以破坏基于梯度或基于编辑的攻击所利用的结构模式。最终响应是根据两个轨道的安全一致性检查来选择的，以确保稳健性，同时不影响响应质量。我们证明，与针对强大对手（例如，DeepIncept，GCG），而不会牺牲良性查询的忠实性。我们的方法始终优于最先进的预算级防御，为更安全的LLM部署提供了实用有效的解决方案。



## **32. Boosting Adversarial Transferability for Hyperspectral Image Classification Using 3D Structure-invariant Transformation and Weighted Intermediate Feature Divergence**

使用3D结构不变变换和加权中间特征分歧提高高光谱图像分类的对抗可移植性 cs.CV

**SubmitDate**: 2025-08-19    [abs](http://arxiv.org/abs/2506.10459v2) [paper-pdf](http://arxiv.org/pdf/2506.10459v2)

**Authors**: Chun Liu, Bingqian Zhu, Tao Xu, Zheng Zheng, Zheng Li, Wei Yang, Zhigang Han, Jiayao Wang

**Abstract**: Deep Neural Networks (DNNs) are vulnerable to adversarial attacks, which pose security challenges to hyperspectral image (HSI) classification based on DNNs. Numerous adversarial attack methods have been designed in the domain of natural images. However, different from natural images, HSIs contains high-dimensional rich spectral information, which presents new challenges for generating adversarial examples. Based on the specific characteristics of HSIs, this paper proposes a novel method to enhance the transferability of the adversarial examples for HSI classification using 3D structure-invariant transformation and weighted intermediate feature divergence. While keeping the HSIs structure invariant, the proposed method divides the image into blocks in both spatial and spectral dimensions. Then, various transformations are applied on each block to increase input diversity and mitigate the overfitting to substitute models. Moreover, a weighted intermediate feature divergence loss is also designed by leveraging the differences between the intermediate features of original and adversarial examples. It constrains the perturbation direction by enlarging the feature maps of the original examples, and assigns different weights to different feature channels to destroy the features that have a greater impact on HSI classification. Extensive experiments demonstrate that the adversarial examples generated by the proposed method achieve more effective adversarial transferability on three public HSI datasets. Furthermore, the method maintains robust attack performance even under defense strategies.

摘要: 深度神经网络（DNN）容易受到对抗攻击，这对基于DNN的高光谱图像（HSI）分类构成了安全挑战。自然图像领域已经设计了许多对抗攻击方法。然而，与自然图像不同，HS包含多维丰富的光谱信息，这为生成对抗性示例提出了新的挑战。根据HSI的具体特征，提出了一种利用3D结构不变变换和加权中间特征分歧来增强HSI分类对抗样本的可移植性的新方法。在保持HS结构不变的同时，提出的方法将图像在空间和光谱维度上划分为块。然后，对每个块应用各种转换，以增加输入多样性并减轻对替代模型的过度适应。此外，还通过利用原始示例和对抗示例的中间特征之间的差异来设计加权中间特征分歧损失。它通过放大原始示例的特征图来约束扰动方向，并为不同的特征通道赋予不同的权重，以破坏对HSI分类影响较大的特征。大量实验表明，该方法生成的对抗示例在三个公共HSI数据集上实现了更有效的对抗可移植性。此外，即使在防御策略下，该方法也能保持稳健的攻击性能。



## **33. A robust and composable device-independent protocol for oblivious transfer using (fully) untrusted quantum devices in the bounded storage model**

有界存储模型下一个鲁棒的可组合的设备无关的不经意传输协议 quant-ph

Major improvement in the main result (security against non-IID  devices)

**SubmitDate**: 2025-08-19    [abs](http://arxiv.org/abs/2404.11283v3) [paper-pdf](http://arxiv.org/pdf/2404.11283v3)

**Authors**: Rishabh Batra, Sayantan Chakraborty, Rahul Jain, Upendra Kapshikar

**Abstract**: We present a robust and composable device-independent (DI) quantum protocol between two parties for oblivious transfer (OT) using Magic Square devices in the bounded storage model in which the (honest and cheating) devices and parties have no long-term quantum memory. After a fixed constant (real-world) time interval, referred to as DELAY, the quantum states decohere completely. The adversary (cheating party), with full control over the devices, is allowed joint (non-IID) quantum operations on the devices, and there are no time and space complexity bounds placed on its powers. The running time of the honest parties is polylog({\lambda}) (where {\lambda} is the security parameter). Our protocol has negligible (in {\lambda}) correctness and security errors and can be implemented in the NISQ (Noisy Intermediate Scale Quantum) era. By robustness, we mean that our protocol is correct even when devices are slightly off (by a small constant) from their ideal specification. This is an important property since small manufacturing errors in the real-world devices are inevitable. Our protocol is sequentially composable and, hence, can be used as a building block to construct larger protocols (including DI bit-commitment and DI secure multi-party computation) while still preserving correctness and security guarantees.   None of the known DI protocols for OT in the literature are robust and secure against joint quantum attacks. This was a major open question in device-independent two-party distrustful cryptography, which we resolve.   We prove a parallel repetition theorem for a certain class of entangled games with a hybrid (quantum-classical) strategy to show the security of our protocol. The hybrid strategy helps to incorporate DELAY in our protocol. This parallel repetition theorem is a main technical contribution of our work.

摘要: 我们在有界存储模型中使用Magic Square设备，在双方之间提出了一种鲁棒且可组合的设备无关（DI）量子协议，在该模型中，（诚实和作弊的）设备和各方没有长期量子记忆。经过固定的恒定（现实世界）时间间隔（称为延迟）后，量子状态完全去散。对手（作弊方）完全控制设备，允许对设备进行联合（非IID）量子操作，并且对其权力没有时间和空间复杂性限制。诚实各方的运行时间是Polylog（{\ambda}）（其中{\ambda}是安全参数）。我们的协议的正确性和安全错误可以忽略不计（在{\ambda}中），并且可以在NISQ（Noisy Intermediate Scale Quantum）时代实施。所谓稳健性，我们的意思是，即使设备与其理想规格略有偏差（一个小的常数），我们的协议也是正确的。这是一个重要的属性，因为现实世界设备中的微小制造错误是不可避免的。我们的协议是可顺序组合的，因此可以用作构建更大协议（包括DI位承诺和DI安全多方计算）的构建模块，同时仍然保留正确性和安全保证。   文献中已知的OT DI协议都不是针对联合量子攻击的稳健且安全的。这是我们解决的与设备无关的双方不信任加密技术中的一个主要悬而未决的问题。   我们用混合（量子经典）策略证明了一类纠缠游戏的并行重复定理，以表明我们协议的安全性。混合策略有助于将DELAY纳入我们的协议中。这个平行重复定理是我们工作的主要技术贡献。



## **34. A Risk Manager for Intrusion Tolerant Systems: Enhancing HAL 9000 with New Scoring and Data Sources**

入侵容忍系统的风险管理器：利用新的评分和数据源增强HAL 9000 cs.CR

**SubmitDate**: 2025-08-18    [abs](http://arxiv.org/abs/2508.13364v1) [paper-pdf](http://arxiv.org/pdf/2508.13364v1)

**Authors**: Tadeu Freitas, Carlos Novo, Inês Dutra, João Soares, Manuel Correia, Benham Shariati, Rolando Martins

**Abstract**: Intrusion Tolerant Systems (ITSs) have become increasingly critical due to the rise of multi-domain adversaries exploiting diverse attack surfaces. ITS architectures aim to tolerate intrusions, ensuring system compromise is prevented or mitigated even with adversary presence. Existing ITS solutions often employ Risk Managers leveraging public security intelligence to adjust system defenses dynamically against emerging threats. However, these approaches rely heavily on databases like NVD and ExploitDB, which require manual analysis for newly discovered vulnerabilities. This dependency limits the system's responsiveness to rapidly evolving threats. HAL 9000, an ITS Risk Manager introduced in our prior work, addressed these challenges through machine learning. By analyzing descriptions of known vulnerabilities, HAL 9000 predicts and assesses new vulnerabilities automatically. To calculate the risk of a system, it also incorporates the Exploitability Probability Scoring system to estimate the likelihood of exploitation within 30 days, enhancing proactive defense capabilities.   Despite its success, HAL 9000's reliance on NVD and ExploitDB knowledge is a limitation, considering the availability of other sources of information. This extended work introduces a custom-built scraper that continuously mines diverse threat sources, including security advisories, research forums, and real-time exploit proofs-of-concept. This significantly expands HAL 9000's intelligence base, enabling earlier detection and assessment of unverified vulnerabilities. Our evaluation demonstrates that integrating scraper-derived intelligence with HAL 9000's risk management framework substantially improves its ability to address emerging threats. This paper details the scraper's integration into the architecture, its role in providing additional information on new threats, and the effects on HAL 9000's management.

摘要: 由于利用不同攻击面的多域对手的兴起，入侵容忍系统（ITS）变得越来越重要。ITS架构旨在容忍入侵，确保即使存在对手也能防止或减轻系统损害。现有的ITS解决方案通常雇用风险经理，利用公共安全情报来动态调整系统防御，以应对新出现的威胁。然而，这些方法严重依赖NVD和ExploitDB等数据库，这些数据库需要对新发现的漏洞进行手动分析。这种依赖性限制了系统对快速变化的威胁的响应能力。HAL 9000是我们在之前的工作中引入的ITS风险管理器，它通过机器学习解决了这些挑战。通过分析已知漏洞的描述，HAL 9000自动预测和评估新的漏洞。为了计算系统的风险，它还集成了可利用性概率评分系统，以估计30天内被利用的可能性，从而增强主动防御能力。   尽管取得了成功，但考虑到其他信息来源的可用性，HAL 9000对NVD和ExploitDB知识的依赖是一个限制。这项扩展工作引入了一个定制的抓取器，可以持续挖掘不同的威胁源，包括安全建议、研究论坛和实时利用概念验证。这显着扩展了HAL 9000的情报基础，能够早期检测和评估未经验证的漏洞。我们的评估表明，将Scrapper衍生的情报与HAL 9000的风险管理框架集成可以极大地提高其应对新兴威胁的能力。本文详细介绍了刮刀的集成到体系结构中，它在提供新威胁的额外信息中的作用，以及对HAL 9000管理的影响。



## **35. Augmented Adversarial Trigger Learning**

增强对抗触发学习 cs.LG

Findings of the Association for Computational Linguistics: NAACL 2025

**SubmitDate**: 2025-08-18    [abs](http://arxiv.org/abs/2503.12339v3) [paper-pdf](http://arxiv.org/pdf/2503.12339v3)

**Authors**: Zhe Wang, Yanjun Qi

**Abstract**: Gradient optimization-based adversarial attack methods automate the learning of adversarial triggers to generate jailbreak prompts or leak system prompts. In this work, we take a closer look at the optimization objective of adversarial trigger learning and propose ATLA: Adversarial Trigger Learning with Augmented objectives. ATLA improves the negative log-likelihood loss used by previous studies into a weighted loss formulation that encourages the learned adversarial triggers to optimize more towards response format tokens. This enables ATLA to learn an adversarial trigger from just one query-response pair and the learned trigger generalizes well to other similar queries. We further design a variation to augment trigger optimization with an auxiliary loss that suppresses evasive responses. We showcase how to use ATLA to learn adversarial suffixes jailbreaking LLMs and to extract hidden system prompts. Empirically we demonstrate that ATLA consistently outperforms current state-of-the-art techniques, achieving nearly 100% success in attacking while requiring 80% fewer queries. ATLA learned jailbreak suffixes demonstrate high generalization to unseen queries and transfer well to new LLMs. We released our code https://github.com/QData/ALTA_Augmented_Adversarial_Trigger_Learning

摘要: 基于梯度优化的对抗攻击方法自动学习对抗触发器，以生成越狱提示或泄露系统提示。在这项工作中，我们仔细研究了对抗性触发学习的优化目标，并提出了ATLA：具有增强目标的对抗性触发学习。ATLA将之前研究使用的负对似然损失改进为加权损失公式，该公式鼓励学习的对抗触发因素对响应格式代币进行更多优化。这使得ATLA能够仅从一个查询-响应对中学习对抗触发器，并且学习到的触发器可以很好地推广到其他类似的查询。我们进一步设计了一个变体，以增加触发优化与辅助损失，抑制逃避反应。我们展示了如何使用ATLA来学习对抗性后缀、越狱LLM和提取隐藏的系统提示。从经验上讲，我们证明，ATLA始终优于当前最先进的技术，实现了近100%的成功攻击，同时需要减少80%的查询。ATLA学习的越狱后缀对看不见的查询具有很高的泛化能力，并可以很好地转移到新的LLM。我们发布了我们的代码https://github.com/QData/ALTA_Augmented_Adversarial_Trigger_Learning



## **36. DAASH: A Meta-Attack Framework for Synthesizing Effective and Stealthy Adversarial Examples**

DAASH：一个用于合成有效且隐蔽的对抗示例的元攻击框架 cs.CV

**SubmitDate**: 2025-08-18    [abs](http://arxiv.org/abs/2508.13309v1) [paper-pdf](http://arxiv.org/pdf/2508.13309v1)

**Authors**: Abdullah Al Nomaan Nafi, Habibur Rahaman, Zafaryab Haider, Tanzim Mahfuz, Fnu Suya, Swarup Bhunia, Prabuddha Chakraborty

**Abstract**: Numerous techniques have been proposed for generating adversarial examples in white-box settings under strict Lp-norm constraints. However, such norm-bounded examples often fail to align well with human perception, and only recently have a few methods begun specifically exploring perceptually aligned adversarial examples. Moreover, it remains unclear whether insights from Lp-constrained attacks can be effectively leveraged to improve perceptual efficacy. In this paper, we introduce DAASH, a fully differentiable meta-attack framework that generates effective and perceptually aligned adversarial examples by strategically composing existing Lp-based attack methods. DAASH operates in a multi-stage fashion: at each stage, it aggregates candidate adversarial examples from multiple base attacks using learned, adaptive weights and propagates the result to the next stage. A novel meta-loss function guides this process by jointly minimizing misclassification loss and perceptual distortion, enabling the framework to dynamically modulate the contribution of each base attack throughout the stages. We evaluate DAASH on adversarially trained models across CIFAR-10, CIFAR-100, and ImageNet. Despite relying solely on Lp-constrained based methods, DAASH significantly outperforms state-of-the-art perceptual attacks such as AdvAD -- achieving higher attack success rates (e.g., 20.63\% improvement) and superior visual quality, as measured by SSIM, LPIPS, and FID (improvements $\approx$ of 11, 0.015, and 5.7, respectively). Furthermore, DAASH generalizes well to unseen defenses, making it a practical and strong baseline for evaluating robustness without requiring handcrafted adaptive attacks for each new defense.

摘要: 人们提出了许多技术来在严格的LP规范约束下在白盒环境中生成对抗性示例。然而，此类规范有界的例子往往无法与人类的感知很好地一致，直到最近才有一些方法开始专门探索感知一致的对抗性例子。此外，目前尚不清楚来自LP约束攻击的见解是否可以有效地利用来提高感知功效。本文中，我们介绍了DAASH，这是一个完全可区分的元攻击框架，通过战略性地组合现有的基于LP的攻击方法来生成有效且感知一致的对抗示例。DAASH以多阶段的方式运行：在每个阶段，它使用学习的自适应权重聚合来自多个基础攻击的候选对抗示例，并将结果传播到下一阶段。一种新颖的元损失函数通过联合最大限度地减少误分类损失和感知失真来指导这一过程，使框架能够动态调节整个阶段每个碱基攻击的贡献。我们在CIFAR-10、CIFAR-100和ImageNet中的对抗训练模型上评估DAASH。尽管仅依赖于基于LP约束的方法，DAASH的性能显着优于AdvAD等最先进的感知攻击--实现更高的攻击成功率（例如，改善20.63%）和优异的视觉质量，通过SSIM、LPIPS和DID衡量（改善$\约为11、0.015和5.7）。此外，DAASH很好地推广到了看不见的防御，使其成为评估稳健性的实用且强大的基线，而无需对每个新防御进行手工设计的自适应攻击。



## **37. RoTO: Robust Topology Obfuscation Against Tomography Inference Attacks**

RoTO：针对断层摄影推断攻击的鲁棒性拓扑混淆 cs.NI

**SubmitDate**: 2025-08-18    [abs](http://arxiv.org/abs/2508.12852v1) [paper-pdf](http://arxiv.org/pdf/2508.12852v1)

**Authors**: Chengze Du, Heng Xu, Zhiwei Yu, Ying Zhou, Zili Meng, Jialong Li

**Abstract**: Tomography inference attacks aim to reconstruct network topology by analyzing end-to-end probe delays. Existing defenses mitigate these attacks by manipulating probe delays to mislead inference, but rely on two strong assumptions: (i) probe packets can be perfectly detected and altered, and (ii) attackers use known, fixed inference algorithms. These assumptions often break in practice, leading to degraded defense performance under detection errors or adaptive adversaries. We present RoTO, a robust topology obfuscation scheme that eliminates both assumptions by modeling uncertainty in attacker-observed delays through a distributional formulation. RoTO casts the defense objective as a min-max optimization problem that maximizes expected topological distortion across this uncertainty set, without relying on perfect probe control or specific attacker models. To approximate attacker behavior, RoTO leverages graph neural networks for inference simulation and adversarial training. We also derive an upper bound on attacker success probability, and demonstrate that our approach enhances topology obfuscation performance through the optimization of this upper bound. Experimental results show that RoTO outperforms existing defense methods, achieving average improvements of 34% in structural similarity and 42.6% in link distance while maintaining strong robustness and concealment capabilities.

摘要: 断层扫描推理攻击旨在通过分析端到端探测延迟来重建网络布局。现有的防御措施通过操纵探测延迟来误导推理来减轻这些攻击，但依赖于两个强假设：（i）探测数据包可以被完美检测和更改，（ii）攻击者使用已知的固定推理算法。这些假设在实践中经常被打破，导致在检测错误或自适应对手的情况下防御性能下降。我们提出了RoTO，这是一种稳健的布局混淆方案，通过分布式公式对攻击者观察到的延迟中的不确定性进行建模，消除了这两种假设。RoTO将防御目标视为一个最小-最大优化问题，该问题最大化了这个不确定性集中的预期拓扑失真，而不依赖于完美的探测控制或特定的攻击者模型。为了逼近攻击者的行为，RoTO利用图神经网络进行推理模拟和对抗训练。我们还推导出攻击者成功概率的上界，并证明我们的方法通过优化该上界来增强了拓扑混淆性能。实验结果表明，RoTO优于现有防御方法，结构相似度平均提高34%，链接距离平均提高42.6%，同时保持了较强的鲁棒性和隐藏能力。



## **38. Deep Positive-Negative Prototypes for Adversarially Robust Discriminative Prototypical Learning**

用于对抗稳健区分原型学习的深度正负原型 cs.LG

This version substantially revises the manuscript, including a new  title and updated experimental results

**SubmitDate**: 2025-08-18    [abs](http://arxiv.org/abs/2504.03782v2) [paper-pdf](http://arxiv.org/pdf/2504.03782v2)

**Authors**: Ramin Zarei Sabzevar, Hamed Mohammadzadeh, Tahmineh Tavakoli, Ahad Harati

**Abstract**: Despite the advantages of discriminative prototype-based methods, their role in adversarial robustness remains underexplored. Meanwhile, current adversarial training methods predominantly focus on robustness against adversarial attacks without explicitly leveraging geometric structures in the latent space, usually resulting in reduced accuracy on the original clean data. We propose a novel framework named Adversarially trained Deep Positive-Negative Prototypes (Adv-DPNP), which integrates discriminative prototype-based learning with adversarial training. Adv-DPNP uses unified class prototypes that serve as both classifier weights and robust anchors in the latent space. Moreover, a novel dual-branch training mechanism maintains stable prototypes by updating them exclusively with clean data, while the feature extractor is trained on both clean and adversarial inputs to increase invariance to adversarial perturbations. In addition, we use a composite loss that combines positive-prototype alignment, negative-prototype repulsion, and consistency regularization to further enhance discrimination, adversarial robustness, and clean accuracy. Extensive experiments on standard benchmarks (CIFAR-10/100 and SVHN) confirm that Adv-DPNP improves clean accuracy over state-of-the-art defenses and baseline methods, while maintaining competitive or superior robustness under a suite of widely used attacks, including FGSM, PGD, C\&W, and AutoAttack. We also evaluate robustness to common corruptions on CIFAR-10-C, where Adv-DPNP achieves the highest average accuracy across severities and corruption types. Additionally, we provide an in-depth analysis of the discriminative quality of the learned feature representations, highlighting the effectiveness of Adv-DPNP in maintaining compactness and clear separation in the latent space.

摘要: 尽管基于区分原型的方法具有优势，但它们在对抗稳健性中的作用仍然没有得到充分的研究。与此同时，当前的对抗性训练方法主要关注针对对抗性攻击的鲁棒性，而没有明确利用潜在空间中的几何结构，这通常会导致原始干净数据的准确性降低。我们提出了一个名为对抗训练的深度正负原型（Adv-DPNP）的新颖框架，它将基于区分原型的学习与对抗训练集成在一起。Adv-DPNP使用统一的类原型，作为潜在空间中的分类器权重和稳健锚点。此外，一种新型的双分支训练机制通过专门使用干净数据更新原型来维护稳定的原型，而特征提取器则在干净和对抗性输入上进行训练，以增加对抗性扰动的不变性。此外，我们使用结合了正原型对齐、负原型排斥和一致性正规化的复合损失，以进一步增强区分力、对抗鲁棒性和清晰准确性。对标准基准测试（CIFAR-10/100和SVHN）的广泛实验证实，Adv-DPNP比最先进的防御和基线方法提高了清晰的准确性，同时在一系列广泛使用的攻击（包括FGSM、PGD、C\& W和AutoAttack）下保持有竞争力或卓越的鲁棒性。我们还评估CIFAR-10-C上对常见腐败的稳健性，其中Adv-DPNP在严重程度和腐败类型方面实现了最高的平均准确性。此外，我们还对学习特征表示的区分质量进行了深入分析，强调了Adv-DPNP在保持潜在空间的紧凑性和清晰分离方面的有效性。



## **39. Boosting Active Defense Persistence: A Two-Stage Defense Framework Combining Interruption and Poisoning Against Deepfake**

提高主动防御持久性：结合中断和毒害来对抗Deepfake的两阶段防御框架 cs.CV

**SubmitDate**: 2025-08-18    [abs](http://arxiv.org/abs/2508.07795v2) [paper-pdf](http://arxiv.org/pdf/2508.07795v2)

**Authors**: Hongrui Zheng, Yuezun Li, Liejun Wang, Yunfeng Diao, Zhiqing Guo

**Abstract**: Active defense strategies have been developed to counter the threat of deepfake technology. However, a primary challenge is their lack of persistence, as their effectiveness is often short-lived. Attackers can bypass these defenses by simply collecting protected samples and retraining their models. This means that static defenses inevitably fail when attackers retrain their models, which severely limits practical use. We argue that an effective defense not only distorts forged content but also blocks the model's ability to adapt, which occurs when attackers retrain their models on protected images. To achieve this, we propose an innovative Two-Stage Defense Framework (TSDF). Benefiting from the intensity separation mechanism designed in this paper, the framework uses dual-function adversarial perturbations to perform two roles. First, it can directly distort the forged results. Second, it acts as a poisoning vehicle that disrupts the data preparation process essential for an attacker's retraining pipeline. By poisoning the data source, TSDF aims to prevent the attacker's model from adapting to the defensive perturbations, thus ensuring the defense remains effective long-term. Comprehensive experiments show that the performance of traditional interruption methods degrades sharply when it is subjected to adversarial retraining. However, our framework shows a strong dual defense capability, which can improve the persistence of active defense. Our code will be available at https://github.com/vpsg-research/TSDF.

摘要: 主动防御策略已经被开发出来来应对深度伪造技术的威胁。然而，主要挑战是它们缺乏持久性，因为它们的有效性往往是短暂的。攻击者只需收集受保护的样本并重新训练他们的模型就可以绕过这些防御。这意味着当攻击者重新训练其模型时，静态防御不可避免地会失败，这严重限制了实际使用。我们认为，有效的防御不仅会扭曲伪造的内容，还会阻止模型的适应能力，当攻击者在受保护的图像上重新训练他们的模型时，就会发生这种情况。为了实现这一目标，我们提出了一个创新的两阶段防御框架（TSDF）。受益于本文设计的强度分离机制，该框架使用双功能对抗扰动来执行两个角色。首先，它可以直接扭曲伪造的结果。其次，它充当投毒工具，扰乱攻击者再培训管道至关重要的数据准备过程。通过毒害数据源，TSDF旨在防止攻击者的模型适应防御扰动，从而确保防御保持长期有效。综合实验表明，传统中断方法在经历对抗性再训练时，性能会急剧下降。但我们的框架表现出强大的双重防御能力，可以提高主动防御的持久性。我们的代码可在https://github.com/vpsg-research/TSDF上获取。



## **40. Concealment of Intent: A Game-Theoretic Analysis**

意图的隐瞒：游戏理论分析 cs.CL

**SubmitDate**: 2025-08-18    [abs](http://arxiv.org/abs/2505.20841v2) [paper-pdf](http://arxiv.org/pdf/2505.20841v2)

**Authors**: Xinbo Wu, Abhishek Umrawal, Lav R. Varshney

**Abstract**: As large language models (LLMs) grow more capable, concerns about their safe deployment have also grown. Although alignment mechanisms have been introduced to deter misuse, they remain vulnerable to carefully designed adversarial prompts. In this work, we present a scalable attack strategy: intent-hiding adversarial prompting, which conceals malicious intent through the composition of skills. We develop a game-theoretic framework to model the interaction between such attacks and defense systems that apply both prompt and response filtering. Our analysis identifies equilibrium points and reveals structural advantages for the attacker. To counter these threats, we propose and analyze a defense mechanism tailored to intent-hiding attacks. Empirically, we validate the attack's effectiveness on multiple real-world LLMs across a range of malicious behaviors, demonstrating clear advantages over existing adversarial prompting techniques.

摘要: 随着大型语言模型（LLM）变得越来越强大，对其安全部署的担忧也越来越多。虽然已经引入了协调机制以防止滥用，但它们仍然容易受到精心设计的对抗性提示的影响。在这项工作中，我们提出了一个可扩展的攻击策略：意图隐藏对抗性提示，通过技能的组合隐藏恶意意图。我们开发了一个博弈论框架来模拟这种攻击和防御系统，适用于即时和响应过滤之间的相互作用。我们的分析确定了平衡点并揭示了攻击者的结构优势。为了应对这些威胁，我们提出并分析了一种针对意图隐藏攻击的防御机制。从经验上讲，我们验证了攻击对一系列恶意行为的多个现实世界LLM的有效性，展示了比现有对抗提示技术的明显优势。



## **41. Quantifying Loss Aversion in Cyber Adversaries via LLM Analysis**

通过LLM分析量化网络对手的损失厌恶 cs.CR

**SubmitDate**: 2025-08-18    [abs](http://arxiv.org/abs/2508.13240v1) [paper-pdf](http://arxiv.org/pdf/2508.13240v1)

**Authors**: Soham Hans, Nikolos Gurney, Stacy Marsella, Sofia Hirschmann

**Abstract**: Understanding and quantifying human cognitive biases from empirical data has long posed a formidable challenge, particularly in cybersecurity, where defending against unknown adversaries is paramount. Traditional cyber defense strategies have largely focused on fortification, while some approaches attempt to anticipate attacker strategies by mapping them to cognitive vulnerabilities, yet they fall short in dynamically interpreting attacks in progress. In recognition of this gap, IARPA's ReSCIND program seeks to infer, defend against, and even exploit attacker cognitive traits. In this paper, we present a novel methodology that leverages large language models (LLMs) to extract quantifiable insights into the cognitive bias of loss aversion from hacker behavior. Our data are collected from an experiment in which hackers were recruited to attack a controlled demonstration network. We process the hacker generated notes using LLMs using it to segment the various actions and correlate the actions to predefined persistence mechanisms used by hackers. By correlating the implementation of these mechanisms with various operational triggers, our analysis provides new insights into how loss aversion manifests in hacker decision-making. The results demonstrate that LLMs can effectively dissect and interpret nuanced behavioral patterns, thereby offering a transformative approach to enhancing cyber defense strategies through real-time, behavior-based analysis.

摘要: 从经验数据中了解和量化人类认知偏差长期以来一直构成一个巨大的挑战，特别是在网络安全领域，防御未知对手至关重要。传统的网络防御策略主要集中在防御上，而一些方法试图通过将攻击者策略映射到认知漏洞来预测攻击者策略，但它们在动态解释正在进行的攻击方面存在缺陷。认识到这一差距，IARPA的ReSCIND计划试图推断、防御甚至利用攻击者的认知特征。在本文中，我们提出了一种新颖的方法，该方法利用大型语言模型（LLM）来提取对黑客行为损失厌恶的认知偏差的可量化见解。我们的数据是从招募黑客攻击受控演示网络的实验中收集的。我们使用LLM处理黑客生成的笔记，使用它来分割各种操作，并将这些操作与黑客使用的预定义的持久性机制关联起来。通过将这些机制的实施与各种操作触发器关联起来，我们的分析为损失厌恶如何体现在黑客决策中提供了新的见解。结果表明，LLM可以有效地剖析和解释细微差别的行为模式，从而提供一种变革性的方法，通过实时、基于行为的分析来增强网络防御策略。



## **42. Heuristic-Induced Multimodal Risk Distribution Jailbreak Attack for Multimodal Large Language Models**

启发式多峰大语言模型的多峰风险分布越狱攻击 cs.CR

ICCV 2025

**SubmitDate**: 2025-08-18    [abs](http://arxiv.org/abs/2412.05934v3) [paper-pdf](http://arxiv.org/pdf/2412.05934v3)

**Authors**: Ma Teng, Jia Xiaojun, Duan Ranjie, Li Xinfeng, Huang Yihao, Jia Xiaoshuang, Chu Zhixuan, Ren Wenqi

**Abstract**: With the rapid advancement of multimodal large language models (MLLMs), concerns regarding their security have increasingly captured the attention of both academia and industry. Although MLLMs are vulnerable to jailbreak attacks, designing effective jailbreak attacks poses unique challenges, especially given the highly constrained adversarial capabilities in real-world deployment scenarios. Previous works concentrate risks into a single modality, resulting in limited jailbreak performance. In this paper, we propose a heuristic-induced multimodal risk distribution jailbreak attack method, called HIMRD, which is black-box and consists of two elements: multimodal risk distribution strategy and heuristic-induced search strategy. The multimodal risk distribution strategy is used to distribute harmful semantics into multiple modalities to effectively circumvent the single-modality protection mechanisms of MLLMs. The heuristic-induced search strategy identifies two types of prompts: the understanding-enhancing prompt, which helps MLLMs reconstruct the malicious prompt, and the inducing prompt, which increases the likelihood of affirmative outputs over refusals, enabling a successful jailbreak attack. HIMRD achieves an average attack success rate (ASR) of 90% across seven open-source MLLMs and an average ASR of around 68% in three closed-source MLLMs. HIMRD reveals cross-modal security vulnerabilities in current MLLMs and underscores the imperative for developing defensive strategies to mitigate such emerging risks. Code is available at https://github.com/MaTengSYSU/HIMRD-jailbreak.

摘要: 随着多模式大型语言模型（MLLM）的快速发展，对其安全性的担忧越来越引起学术界和工业界的关注。尽管MLLM容易受到越狱攻击，但设计有效的越狱攻击带来了独特的挑战，特别是考虑到现实世界部署场景中的对抗能力受到高度限制。之前的作品将风险集中在单一模式中，导致越狱表现有限。本文提出了一种启发式多峰风险分布越狱攻击方法，称为HIMRD，它是黑匣子，由两个元素组成：多峰风险分布策略和启发式搜索策略。多模式风险分布策略用于将有害语义分布到多个模式中，以有效规避MLLM的单模式保护机制。启发式搜索策略识别了两种类型的提示：增强理解提示，帮助MLLM重建恶意提示，以及诱导提示，增加了肯定输出而不是拒绝的可能性，从而实现成功的越狱攻击。HIMRD在七个开源MLLM中实现了90%的平均攻击成功率（ASB），在三个开源MLLM中实现了平均攻击成功率（ASB）约为68%。HIMRD揭示了当前MLLM中的跨模式安全漏洞，并强调制定防御策略以减轻此类新出现的风险的必要性。代码可在https://github.com/MaTengSYSU/HIMRD-jailbreak上获取。



## **43. Reducing False Positives with Active Behavioral Analysis for Cloud Security**

通过云安全的主动行为分析减少误报 cs.CR

**SubmitDate**: 2025-08-18    [abs](http://arxiv.org/abs/2508.12584v1) [paper-pdf](http://arxiv.org/pdf/2508.12584v1)

**Authors**: Dikshant, Verma

**Abstract**: Rule-based cloud security posture management (CSPM) solutions are known to produce a lot of false positives based on the limited contextual understanding and dependence on static heuristics testing. This paper introduces a validation-driven methodology that integrates active behavioral testing in cloud security posture management solution(s) to evaluate the exploitability of policy violations in real time. The proposed system employs lightweight and automated probes, built from open-source tools, validation scripts, and penetration testing test cases, to simulate adversarial attacks on misconfigured or vulnerable cloud assets without any impact to the cloud services or environment. For instance, cloud services may be flagged as publicly exposed and vulnerable despite being protected by access control layers, or secure policies, resulting in non-actionable alerts that consumes analysts time during manual validation. Through controlled experimentation in a reproducible AWS setup, we evaluated the reduction in false positive rates across various misconfiguration and vulnerable alerts. Our findings indicate an average reduction of 93\% in false positives. Furthermore, the framework demonstrates low latency performance. These results demonstrate a scalable method to improve detection accuracy and analyst productivity in large cloud environments. While our evaluation focuses on AWS, the architecture is modular and extensible to multi-cloud setups.

摘要: 众所周知，基于规则的云安全态势管理（CSPP）解决方案会基于有限的上下文理解和对静态启发式测试的依赖而产生大量误报。本文介绍了一种验证驱动的方法，该方法将主动行为测试集成到云安全态势管理解决方案中，以实时评估策略违规的利用性。拟议的系统采用轻量级和自动化的探测器，由开源工具、验证脚本和渗透测试测试用例构建，来模拟对配置错误或脆弱的云资产的对抗攻击，而不会对云服务或环境产生任何影响。例如，尽管受到访问控制层或安全策略的保护，云服务仍可能被标记为公开暴露且易受攻击，从而导致不可操作的警报，从而在手动验证期间消耗分析师的时间。通过在可重复的AWS设置中进行受控实验，我们评估了各种错误配置和漏洞警报中假阳性率的降低情况。我们的研究结果表明假阳性平均减少了93%。此外，该框架展示了低延迟性能。这些结果展示了一种可扩展的方法，可以在大型云环境中提高检测准确性和分析师生产力。虽然我们的评估重点关注AWS，但该架构是模块化的，可扩展到多云设置。



## **44. Adversarial Attacks on VQA-NLE: Exposing and Alleviating Inconsistencies in Visual Question Answering Explanations**

对VQA-NLE的对抗性攻击：暴露和减轻视觉问题回答解释中的不确定性 cs.CV

**SubmitDate**: 2025-08-17    [abs](http://arxiv.org/abs/2508.12430v1) [paper-pdf](http://arxiv.org/pdf/2508.12430v1)

**Authors**: Yahsin Yeh, Yilun Wu, Bokai Ruan, Honghan Shuai

**Abstract**: Natural language explanations in visual question answering (VQA-NLE) aim to make black-box models more transparent by elucidating their decision-making processes. However, we find that existing VQA-NLE systems can produce inconsistent explanations and reach conclusions without genuinely understanding the underlying context, exposing weaknesses in either their inference pipeline or explanation-generation mechanism. To highlight these vulnerabilities, we not only leverage an existing adversarial strategy to perturb questions but also propose a novel strategy that minimally alters images to induce contradictory or spurious outputs. We further introduce a mitigation method that leverages external knowledge to alleviate these inconsistencies, thereby bolstering model robustness. Extensive evaluations on two standard benchmarks and two widely used VQA-NLE models underscore the effectiveness of our attacks and the potential of knowledge-based defenses, ultimately revealing pressing security and reliability concerns in current VQA-NLE systems.

摘要: 视觉问答中的自然语言解释（VQA-NLE）旨在通过阐明黑匣子模型的决策过程来使其更加透明。然而，我们发现，现有的VQA-NLE系统可能会在不真正理解底层上下文的情况下产生不一致的解释并得出结论，从而暴露了其推理管道或推理生成机制的弱点。为了强调这些漏洞，我们不仅利用现有的对抗策略来扰乱问题，而且还提出了一种新颖的策略，可以最大限度地改变图像以引发矛盾或虚假的输出。我们进一步引入了一种缓解方法，利用外部知识来缓解这些不一致性，从而增强模型的稳健性。对两个标准基准和两个广泛使用的VQA-NLE模型的广泛评估强调了我们攻击的有效性和基于知识的防御的潜力，最终揭示了当前VQA-NLE系统中紧迫的安全性和可靠性问题。



## **45. Cascading and Proxy Membership Inference Attacks**

级联和代理成员推断攻击 cs.CR

Accepted by NDSS 2026

**SubmitDate**: 2025-08-17    [abs](http://arxiv.org/abs/2507.21412v2) [paper-pdf](http://arxiv.org/pdf/2507.21412v2)

**Authors**: Yuntao Du, Jiacheng Li, Yuetian Chen, Kaiyuan Zhang, Zhizhen Yuan, Hanshen Xiao, Bruno Ribeiro, Ninghui Li

**Abstract**: A Membership Inference Attack (MIA) assesses how much a trained machine learning model reveals about its training data by determining whether specific query instances were included in the dataset. We classify existing MIAs into adaptive or non-adaptive, depending on whether the adversary is allowed to train shadow models on membership queries. In the adaptive setting, where the adversary can train shadow models after accessing query instances, we highlight the importance of exploiting membership dependencies between instances and propose an attack-agnostic framework called Cascading Membership Inference Attack (CMIA), which incorporates membership dependencies via conditional shadow training to boost membership inference performance.   In the non-adaptive setting, where the adversary is restricted to training shadow models before obtaining membership queries, we introduce Proxy Membership Inference Attack (PMIA). PMIA employs a proxy selection strategy that identifies samples with similar behaviors to the query instance and uses their behaviors in shadow models to perform a membership posterior odds test for membership inference. We provide theoretical analyses for both attacks, and extensive experimental results demonstrate that CMIA and PMIA substantially outperform existing MIAs in both settings, particularly in the low false-positive regime, which is crucial for evaluating privacy risks.

摘要: 成员资格推理攻击（MIA）通过确定数据集中是否包括特定的查询实例来评估经过训练的机器学习模型对其训练数据的揭示程度。我们将现有的MIA分为自适应或非自适应，具体取决于是否允许对手在成员资格查询上训练影子模型。在自适应环境中，对手可以在访问查询实例后训练影子模型，我们强调了利用实例之间成员依赖关系的重要性，并提出了一种名为级联成员推断攻击（CMIA）的攻击不可知框架，该框架通过条件影子训练合并成员依赖关系，以提高成员推断性能。   在非自适应环境中，对手仅限于在获得成员资格查询之前训练影子模型，我们引入代理成员资格推断攻击（PMIA）。PMIA采用代理选择策略，该策略识别与查询实例具有相似行为的样本，并使用其在影子模型中的行为来执行成员资格后验赔率测试以进行成员资格推断。我们对这两种攻击提供了理论分析，大量的实验结果表明，CMIA和PMIA在这两种环境下的表现都大大优于现有的MIA，特别是在低假阳性机制下，这对于评估隐私风险至关重要。



## **46. ViT-EnsembleAttack: Augmenting Ensemble Models for Stronger Adversarial Transferability in Vision Transformers**

ViT-Ensemble Attack：增强Ensemble模型，以增强Vision Transformers中的对抗可移植性 cs.CV

**SubmitDate**: 2025-08-17    [abs](http://arxiv.org/abs/2508.12384v1) [paper-pdf](http://arxiv.org/pdf/2508.12384v1)

**Authors**: Hanwen Cao, Haobo Lu, Xiaosen Wang, Kun He

**Abstract**: Ensemble-based attacks have been proven to be effective in enhancing adversarial transferability by aggregating the outputs of models with various architectures. However, existing research primarily focuses on refining ensemble weights or optimizing the ensemble path, overlooking the exploration of ensemble models to enhance the transferability of adversarial attacks. To address this gap, we propose applying adversarial augmentation to the surrogate models, aiming to boost overall generalization of ensemble models and reduce the risk of adversarial overfitting. Meanwhile, observing that ensemble Vision Transformers (ViTs) gain less attention, we propose ViT-EnsembleAttack based on the idea of model adversarial augmentation, the first ensemble-based attack method tailored for ViTs to the best of our knowledge. Our approach generates augmented models for each surrogate ViT using three strategies: Multi-head dropping, Attention score scaling, and MLP feature mixing, with the associated parameters optimized by Bayesian optimization. These adversarially augmented models are ensembled to generate adversarial examples. Furthermore, we introduce Automatic Reweighting and Step Size Enlargement modules to boost transferability. Extensive experiments demonstrate that ViT-EnsembleAttack significantly enhances the adversarial transferability of ensemble-based attacks on ViTs, outperforming existing methods by a substantial margin. Code is available at https://github.com/Trustworthy-AI-Group/TransferAttack.

摘要: 基于集合的攻击已被证明可以有效地通过聚合具有各种架构的模型的输出来增强对抗可转移性。然而，现有的研究主要集中在细化集成权重或优化集成路径上，忽视了对集成模型的探索以增强对抗性攻击的可转移性。为了解决这一差距，我们建议对代理模型应用对抗性增强，旨在提高集成模型的总体通用性并降低对抗性过度适应的风险。与此同时，由于注意到整体视觉变形者（ViT）受到的关注较少，我们基于模型对抗增强的想法提出了ViT-EnsembleAttack，这是据我们所知为ViT量身定制的第一种基于整体的攻击方法。我们的方法使用三种策略为每个替代ViT生成增强模型：多头丢弃、注意力分数缩放和MLP特征混合，并通过Bayesian优化优化相关参数。这些对抗增强模型被集成起来以生成对抗示例。此外，我们还引入了自动重新加权和步进大小加入模块以提高可移植性。大量实验表明，ViT-EnsembleAttack显着增强了对ViT的基于集合的攻击的对抗可转移性，大大优于现有方法。代码可在https://github.com/Trustworthy-AI-Group/TransferAttack上获取。



## **47. Jamming Identification with Differential Transformer for Low-Altitude Wireless Networks**

低海拔无线网络中使用差异Transformer的干扰识别 eess.SP

**SubmitDate**: 2025-08-17    [abs](http://arxiv.org/abs/2508.12320v1) [paper-pdf](http://arxiv.org/pdf/2508.12320v1)

**Authors**: Pengyu Wang, Zhaocheng Wang, Tianqi Mao, Weijie Yuan, Haijun Zhang, George K. Karagiannidis

**Abstract**: Wireless jamming identification, which detects and classifies electromagnetic jamming from non-cooperative devices, is crucial for emerging low-altitude wireless networks consisting of many drone terminals that are highly susceptible to electromagnetic jamming. However, jamming identification schemes adopting deep learning (DL) are vulnerable to attacks involving carefully crafted adversarial samples, resulting in inevitable robustness degradation. To address this issue, we propose a differential transformer framework for wireless jamming identification. Firstly, we introduce a differential transformer network in order to distinguish jamming signals, which overcomes the attention noise when compared with its traditional counterpart by performing self-attention operations in a differential manner. Secondly, we propose a randomized masking training strategy to improve network robustness, which leverages the patch partitioning mechanism inherent to transformer architectures in order to create parallel feature extraction branches. Each branch operates on a distinct, randomly masked subset of patches, which fundamentally constrains the propagation of adversarial perturbations across the network. Additionally, the ensemble effect generated by fusing predictions from these diverse branches demonstrates superior resilience against adversarial attacks. Finally, we introduce a novel consistent training framework that significantly enhances adversarial robustness through dualbranch regularization. Simulation results demonstrate that our proposed methodology is superior to existing methods in boosting robustness to adversarial samples.

摘要: 无线干扰识别可检测来自非合作设备的电磁干扰并对其进行分类，对于由许多极易受到电磁干扰的无人机终端组成的新兴低空无线网络至关重要。然而，采用深度学习（DL）的干扰识别方案很容易受到涉及精心设计的对抗样本的攻击，导致不可避免的鲁棒性下降。为了解决这个问题，我们提出了一种用于无线干扰识别的差异Transformer框架。首先，我们引入了一种差异Transformer网络来区分干扰信号，与传统的网络相比，该网络通过以差异方式执行自注意操作来克服了注意噪音。其次，我们提出了一种随机掩蔽训练策略来提高网络稳健性，该策略利用Transformer架构固有的补丁划分机制来创建并行特征提取分支。每个分支对一个不同的、随机屏蔽的补丁子集进行操作，这从根本上限制了对抗性扰动在网络中的传播。此外，融合这些不同分支的预测所产生的综合效应显示了对对抗性攻击的卓越弹性。最后，我们引入了一种新颖的一致训练框架，该框架通过双分支正规化显着增强了对抗鲁棒性。模拟结果表明，我们提出的方法在增强对抗样本的鲁棒性方面优于现有方法。



## **48. Adjustable AprilTags For Identity Secured Tasks**

身份保护任务的可调四月标签 cs.CR

**SubmitDate**: 2025-08-17    [abs](http://arxiv.org/abs/2508.12304v1) [paper-pdf](http://arxiv.org/pdf/2508.12304v1)

**Authors**: Hao Li

**Abstract**: Special tags such as AprilTags that facilitate image processing and pattern recognition are useful in practical applications. In close and private environments, identity security is unlikely to be an issue because all involved AprilTags can be completely regulated. However, in open and public environments, identity security is no longer an issue that can be neglected. To handle potential harm caused by adversarial attacks, this note advocates utilization of adjustable AprilTags instead of fixed ones.

摘要: 促进图像处理和模式识别的特殊标签（例如AprilTags）在实际应用中很有用。在封闭且私密的环境中，身份安全不太可能成为问题，因为所有相关的AprilTags都可以受到完全监管。然而，在开放和公共环境中，身份安全不再是一个可以忽视的问题。为了应对对抗攻击造成的潜在伤害，本说明主张使用可调整的AprilTags而不是固定的AprilTags。



## **49. ForensicsSAM: Toward Robust and Unified Image Forgery Detection and Localization Resisting to Adversarial Attack**

ForensicsSam：迈向稳健、统一的图像伪造检测和定位，抵御对抗攻击 cs.CV

**SubmitDate**: 2025-08-17    [abs](http://arxiv.org/abs/2508.07402v2) [paper-pdf](http://arxiv.org/pdf/2508.07402v2)

**Authors**: Rongxuan Peng, Shunquan Tan, Chenqi Kong, Anwei Luo, Alex C. Kot, Jiwu Huang

**Abstract**: Parameter-efficient fine-tuning (PEFT) has emerged as a popular strategy for adapting large vision foundation models, such as the Segment Anything Model (SAM) and LLaVA, to downstream tasks like image forgery detection and localization (IFDL). However, existing PEFT-based approaches overlook their vulnerability to adversarial attacks. In this paper, we show that highly transferable adversarial images can be crafted solely via the upstream model, without accessing the downstream model or training data, significantly degrading the IFDL performance. To address this, we propose ForensicsSAM, a unified IFDL framework with built-in adversarial robustness. Our design is guided by three key ideas: (1) To compensate for the lack of forgery-relevant knowledge in the frozen image encoder, we inject forgery experts into each transformer block to enhance its ability to capture forgery artifacts. These forgery experts are always activated and shared across any input images. (2) To detect adversarial images, we design an light-weight adversary detector that learns to capture structured, task-specific artifact in RGB domain, enabling reliable discrimination across various attack methods. (3) To resist adversarial attacks, we inject adversary experts into the global attention layers and MLP modules to progressively correct feature shifts induced by adversarial noise. These adversary experts are adaptively activated by the adversary detector, thereby avoiding unnecessary interference with clean images. Extensive experiments across multiple benchmarks demonstrate that ForensicsSAM achieves superior resistance to various adversarial attack methods, while also delivering state-of-the-art performance in image-level forgery detection and pixel-level forgery localization. The resource is available at https://github.com/siriusPRX/ForensicsSAM.

摘要: 参数高效微调（PEFT）已成为一种流行策略，用于将大型视觉基础模型（例如Segment Anything Model（Sam）和LLaVA）适应图像伪造检测和定位（IFDL）等下游任务。然而，现有的基于PEFT的方法忽视了它们对对抗攻击的脆弱性。在本文中，我们表明，高度可转移的对抗图像可以仅通过上游模型制作，而无需访问下游模型或训练数据，从而显着降低IFDL性能。为了解决这个问题，我们提出了ForensicsSam，这是一个具有内置对抗稳健性的统一IFDL框架。我们的设计遵循三个关键思想：（1）为了弥补冻结图像编码器中伪造相关知识的缺乏，我们将伪造专家注入到每个Transformer模块中，以增强其捕获伪造文物的能力。这些伪造专家始终被激活并在任何输入图像中共享。(2)为了检测对抗图像，我们设计了一种轻量级的对手检测器，它可以学习在RB域中捕获结构化的、特定于任务的伪影，从而能够在各种攻击方法之间进行可靠的区分。(3)为了抵抗对抗性攻击，我们将对手专家注入到全局注意力层和MLP模块中，以逐步纠正对抗性噪音引起的特征转变。这些对手专家由对手检测器自适应激活，从而避免对干净图像的不必要干扰。跨多个基准的广泛实验表明，ForensicsSam对各种对抗攻击方法实现了卓越的抵抗力，同时还在图像级伪造检测和像素级伪造定位方面提供了最先进的性能。该资源可访问https://github.com/siriusPRX/ForensicsSAM。



## **50. TriQDef: Disrupting Semantic and Gradient Alignment to Prevent Adversarial Patch Transferability in Quantized Neural Networks**

TriQDef：扰乱语义和梯度对齐以防止量化神经网络中的对抗性补丁可移植性 cs.CV

**SubmitDate**: 2025-08-16    [abs](http://arxiv.org/abs/2508.12132v1) [paper-pdf](http://arxiv.org/pdf/2508.12132v1)

**Authors**: Amira Guesmi, Bassem Ouni, Muhammad Shafique

**Abstract**: Quantized Neural Networks (QNNs) are increasingly deployed in edge and resource-constrained environments due to their efficiency in computation and memory usage. While shown to distort the gradient landscape and weaken conventional pixel-level attacks, it provides limited robustness against patch-based adversarial attacks-localized, high-saliency perturbations that remain surprisingly transferable across bit-widths. Existing defenses either overfit to fixed quantization settings or fail to address this cross-bit generalization vulnerability. We introduce \textbf{TriQDef}, a tri-level quantization-aware defense framework designed to disrupt the transferability of patch-based adversarial attacks across QNNs. TriQDef consists of: (1) a Feature Disalignment Penalty (FDP) that enforces semantic inconsistency by penalizing perceptual similarity in intermediate representations; (2) a Gradient Perceptual Dissonance Penalty (GPDP) that explicitly misaligns input gradients across bit-widths by minimizing structural and directional agreement via Edge IoU and HOG Cosine metrics; and (3) a Joint Quantization-Aware Training Protocol that unifies these penalties within a shared-weight training scheme across multiple quantization levels. Extensive experiments on CIFAR-10 and ImageNet demonstrate that TriQDef reduces Attack Success Rates (ASR) by over 40\% on unseen patch and quantization combinations, while preserving high clean accuracy. Our findings underscore the importance of disrupting both semantic and perceptual gradient alignment to mitigate patch transferability in QNNs.

摘要: 量化神经网络（QNN）由于其计算效率和内存使用效率，越来越多地部署在边缘和资源受限的环境中。虽然它被证明会扭曲梯度景观并削弱传统的像素级攻击，但它对基于补丁的对抗性攻击（局部化的高显着性扰动）提供的鲁棒性有限，这些扰动在跨比特宽度上仍然令人惊讶地可转移。现有的防御要么过度适合固定的量化设置，要么无法解决这种跨位概括漏洞。我们引入了\textBF{TriQDef}，这是一个三级量化感知防御框架，旨在破坏QNN之间基于补丁的对抗性攻击的可转移性。TriQDef由：（1）特征失调罚分（FDP），通过惩罚中间表示中的感知相似性来强制语义不一致;（2）梯度感知失调罚分（GDP），通过最小化经Edge IoU和HOG Cosine度量的结构和方向一致性来明确地在比特宽度上失调输入梯度;以及（3）联合量化感知训练协议，将这些惩罚统一到跨多个量化级别的共享权重训练方案中。CIFAR-10和ImageNet上的大量实验表明，TriQDef在未见补丁和量化组合上将攻击成功率（ASB）降低了40%以上，同时保持了高清晰准确性。我们的研究结果强调了破坏语义和感知梯度对齐以减轻QNN中的补丁可转移性的重要性。



