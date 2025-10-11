# Latest Adversarial Attack Papers
**update at 2025-10-11 09:51:27**

翻译来自 https://cloud.tencent.com/document/product/551/15619

## **1. AutoRed: A Free-form Adversarial Prompt Generation Framework for Automated Red Teaming**

AutoRed：一个用于自动化红色团队的自由形式对抗提示生成框架 cs.CL

**SubmitDate**: 2025-10-09    [abs](http://arxiv.org/abs/2510.08329v1) [paper-pdf](http://arxiv.org/pdf/2510.08329v1)

**Authors**: Muxi Diao, Yutao Mou, Keqing He, Hanbo Song, Lulu Zhao, Shikun Zhang, Wei Ye, Kongming Liang, Zhanyu Ma

**Abstract**: The safety of Large Language Models (LLMs) is crucial for the development of trustworthy AI applications. Existing red teaming methods often rely on seed instructions, which limits the semantic diversity of the synthesized adversarial prompts. We propose AutoRed, a free-form adversarial prompt generation framework that removes the need for seed instructions. AutoRed operates in two stages: (1) persona-guided adversarial instruction generation, and (2) a reflection loop to iteratively refine low-quality prompts. To improve efficiency, we introduce a verifier to assess prompt harmfulness without querying the target models. Using AutoRed, we build two red teaming datasets -- AutoRed-Medium and AutoRed-Hard -- and evaluate eight state-of-the-art LLMs. AutoRed achieves higher attack success rates and better generalization than existing baselines. Our results highlight the limitations of seed-based approaches and demonstrate the potential of free-form red teaming for LLM safety evaluation. We will open source our datasets in the near future.

摘要: 大型语言模型（LLM）的安全性对于开发值得信赖的人工智能应用程序至关重要。现有的红色分组方法通常依赖于种子指令，这限制了合成对抗提示的语义多样性。我们提出AutoRed，这是一个自由形式的对抗性提示生成框架，它消除了对种子指令的需要。AutoRed分两个阶段运行：（1）角色引导的对抗指令生成，和（2）迭代地细化低质量提示的反射循环。为了提高效率，我们引入了一个验证器来评估即时危害性，而无需查询目标模型。使用AutoRed，我们构建了两个红色团队数据集-- AutoRed-Medium和AutoRed-Hard --并评估了八个最先进的LLM。AutoRed比现有基线实现了更高的攻击成功率和更好的概括性。我们的结果强调了基于种子的方法的局限性，并展示了自由形式的红色团队在LLM安全性评估中的潜力。我们将在不久的将来开放我们的数据集。



## **2. Watch your steps: Dormant Adversarial Behaviors that Activate upon LLM Finetuning**

注意步骤：LLM微调后激活的休眠对抗行为 cs.LG

**SubmitDate**: 2025-10-09    [abs](http://arxiv.org/abs/2505.16567v3) [paper-pdf](http://arxiv.org/pdf/2505.16567v3)

**Authors**: Thibaud Gloaguen, Mark Vero, Robin Staab, Martin Vechev

**Abstract**: Finetuning open-weight Large Language Models (LLMs) is standard practice for achieving task-specific performance improvements. Until now, finetuning has been regarded as a controlled and secure process in which training on benign datasets leads to predictable behaviors. In this paper, we demonstrate, for the first time, that an adversary can create compromised LLMs that are performant and benign, yet exhibit adversarial behaviors once finetuned by downstream users. To this end, we propose an attack, FAB (Finetuning-activated Adversarial Behaviors), which compromises an LLM via meta-learning techniques that simulate downstream finetuning, explicitly optimizing for the emergence of adversarial behaviors in the finetuned models. At the same time, the compromised LLM is regularized to retain general capabilities and to exhibit no adversarial behaviors prior to finetuning. As a result, when users finetune (e.g., instruction-tuning, distillation, DPO) the seemingly benign model on their own datasets, they unknowingly trigger its dormant adversarial behavior. We experimentally demonstrate the effectiveness of FAB across multiple LLMs and three commonly considered target behaviors: unsolicited advertising, jailbreakability, and over-refusal. We show that FAB-triggers are robust to various finetuning choices made by the user (e.g., dataset, number of steps, scheduler, post-training algorithm). Our findings challenge prevailing assumptions on the security of finetuning, revealing a critical attack vector.

摘要: 微调开权重大型语言模型（LLM）是实现特定任务性能改进的标准实践。到目前为止，微调一直被认为是一个受控且安全的过程，其中对良性数据集的训练会导致可预测的行为。在本文中，我们首次证明，对手可以创建高性能且良性的受损LLM，但一旦被下游用户微调，就会表现出对抗行为。为此，我们提出了一种攻击FAB（微调激活的对抗行为），它通过模拟下游微调的元学习技术来损害LLM，明确优化微调模型中对抗行为的出现。与此同时，受损的LLM被规范化，以保留一般能力，并且在微调之前不表现出对抗行为。因此，当用户微调（例如，描述-调优、蒸馏、DPO）在他们自己的数据集上看似良性的模型，但他们在不知不觉中触发了其休眠的对抗行为。我们通过实验证明了FAB在多个LLM和三种常见的目标行为中的有效性：未经请求的广告、越狱和过度拒绝。我们表明FAB触发器对用户做出的各种微调选择是稳健的（例如，数据集、步骤数、调度器、训练后算法）。我们的发现挑战了有关微调安全性的普遍假设，揭示了一个关键的攻击载体。



## **3. The Alignment Waltz: Jointly Training Agents to Collaborate for Safety**

一致华尔兹：联合培训特工为安全进行协作 cs.CL

**SubmitDate**: 2025-10-09    [abs](http://arxiv.org/abs/2510.08240v1) [paper-pdf](http://arxiv.org/pdf/2510.08240v1)

**Authors**: Jingyu Zhang, Haozhu Wang, Eric Michael Smith, Sid Wang, Amr Sharaf, Mahesh Pasupuleti, Benjamin Van Durme, Daniel Khashabi, Jason Weston, Hongyuan Zhan

**Abstract**: Harnessing the power of LLMs requires a delicate dance between being helpful and harmless. This creates a fundamental tension between two competing challenges: vulnerability to adversarial attacks that elicit unsafe content, and a tendency for overrefusal on benign but sensitive prompts. Current approaches often navigate this dance with safeguard models that completely reject any content that contains unsafe portions. This approach cuts the music entirely-it may exacerbate overrefusals and fails to provide nuanced guidance for queries it refuses. To teach models a more coordinated choreography, we propose WaltzRL, a novel multi-agent reinforcement learning framework that formulates safety alignment as a collaborative, positive-sum game. WaltzRL jointly trains a conversation agent and a feedback agent, where the latter is incentivized to provide useful suggestions that improve the safety and helpfulness of the conversation agent's responses. At the core of WaltzRL is a Dynamic Improvement Reward (DIR) that evolves over time based on how well the conversation agent incorporates the feedback. At inference time, unsafe or overrefusing responses from the conversation agent are improved rather than discarded. The feedback agent is deployed together with the conversation agent and only engages adaptively when needed, preserving helpfulness and low latency on safe queries. Our experiments, conducted across five diverse datasets, demonstrate that WaltzRL significantly reduces both unsafe responses (e.g., from 39.0% to 4.6% on WildJailbreak) and overrefusals (from 45.3% to 9.9% on OR-Bench) compared to various baselines. By enabling the conversation and feedback agents to co-evolve and adaptively apply feedback, WaltzRL enhances LLM safety without degrading general capabilities, thereby advancing the Pareto front between helpfulness and harmlessness.

摘要: 利用LLM的力量需要在乐于助人和无害之间进行微妙的舞蹈。这在两个相互竞争的挑战之间造成了根本性的紧张关系：容易受到引发不安全内容的对抗攻击，以及在良性但敏感的提示下过度拒绝的倾向。当前的方法通常通过完全拒绝任何包含不安全部分的内容的安全模型来应对这一转变。这种方法完全削弱了音乐--它可能会加剧过度拒绝，并且无法为它拒绝的询问提供细致入微的指导。为了教模型更协调的编排，我们提出了WaltzRL，一种新的多智能体强化学习框架，将安全对齐制定为协作的正和游戏。WaltzRL联合训练会话代理和反馈代理，后者被激励提供有用的建议，以提高会话代理响应的安全性和有用性。WaltzRL的核心是一个动态改进奖励（Dynamic Improvement Reward），它会根据会话代理整合反馈的程度随着时间的推移而演变。在推理时，来自会话代理的不安全或过度拒绝的响应被改进而不是被丢弃。反馈代理与对话代理一起部署，并且仅在需要时自适应地参与，从而在安全查询上保持帮助性和低延迟。我们在五个不同的数据集上进行的实验表明，WaltzRL显着减少了两种不安全的响应（例如，与各种基线相比，WildJailbreak的比例从39.0%上升到4.6%）和过度拒绝（OR-Bench的比例从45.3%上升到9.9%）。通过使对话和反馈代理能够共同进化并自适应地应用反馈，WaltzRL在不降低一般能力的情况下增强了LLM安全性，从而在有益和无害之间推进帕累托战线。



## **4. Multi-Trigger Poisoning Amplifies Backdoor Vulnerabilities in LLMs**

多触发中毒放大了LLM中的后门漏洞 cs.CL

**SubmitDate**: 2025-10-09    [abs](http://arxiv.org/abs/2507.11112v2) [paper-pdf](http://arxiv.org/pdf/2507.11112v2)

**Authors**: Sanhanat Sivapiromrat, Caiqi Zhang, Marco Basaldella, Nigel Collier

**Abstract**: Recent studies have shown that Large Language Models (LLMs) are vulnerable to data poisoning attacks, where malicious training examples embed hidden behaviours triggered by specific input patterns. However, most existing works assume a phrase and focus on the attack's effectiveness, offering limited understanding of trigger mechanisms and how multiple triggers interact within the model. In this paper, we present a framework for studying poisoning in LLMs. We show that multiple distinct backdoor triggers can coexist within a single model without interfering with each other, enabling adversaries to embed several triggers concurrently. Using multiple triggers with high embedding similarity, we demonstrate that poisoned triggers can achieve robust activation even when tokens are substituted or separated by long token spans. Our findings expose a broader and more persistent vulnerability surface in LLMs. To mitigate this threat, we propose a post hoc recovery method that selectively retrains specific model components based on a layer-wise weight difference analysis. Our method effectively removes the trigger behaviour with minimal parameter updates, presenting a practical and efficient defence against multi-trigger poisoning.

摘要: 最近的研究表明，大型语言模型（LLM）很容易受到数据中毒攻击，其中恶意训练示例嵌入了由特定输入模式触发的隐藏行为。然而，大多数现有的作品假设一个短语并关注攻击的有效性，对触发机制以及多个触发如何在模型内相互作用的理解有限。本文中，我们提出了一个研究LLM中毒的框架。我们表明，多个不同的后门触发器可以在单个模型中共存，而不会相互干扰，从而使对手能够同时嵌入多个触发器。使用具有高嵌入相似性的多个触发器，我们证明即使令牌被长令牌跨度替换或分开，中毒触发器也可以实现稳健的激活。我们的研究结果揭示了LLC中更广泛、更持久的脆弱性表面。为了减轻这种威胁，我们提出了一种事后恢复方法，该方法根据分层权重差异分析选择性地重新训练特定的模型组件。我们的方法通过最少的参数更新有效地消除了触发行为，从而提供了针对多触发中毒的实用有效防御。



## **5. Interpreting LLM-as-a-Judge Policies via Verifiable Global Explanations**

通过可验证的全球解释解释法学硕士作为法官的政策 cs.CL

12 pages, 2 figures, 3 tables

**SubmitDate**: 2025-10-09    [abs](http://arxiv.org/abs/2510.08120v1) [paper-pdf](http://arxiv.org/pdf/2510.08120v1)

**Authors**: Jasmina Gajcin, Erik Miehling, Rahul Nair, Elizabeth Daly, Radu Marinescu, Seshu Tirupathi

**Abstract**: Using LLMs to evaluate text, that is, LLM-as-a-judge, is increasingly being used at scale to augment or even replace human annotations. As such, it is imperative that we understand the potential biases and risks of doing so. In this work, we propose an approach for extracting high-level concept-based global policies from LLM-as-a-Judge. Our approach consists of two algorithms: 1) CLoVE (Contrastive Local Verifiable Explanations), which generates verifiable, concept-based, contrastive local explanations and 2) GloVE (Global Verifiable Explanations), which uses iterative clustering, summarization and verification to condense local rules into a global policy. We evaluate GloVE on seven standard benchmarking datasets for content harm detection. We find that the extracted global policies are highly faithful to decisions of the LLM-as-a-Judge. Additionally, we evaluated the robustness of global policies to text perturbations and adversarial attacks. Finally, we conducted a user study to evaluate user understanding and satisfaction with global policies.

摘要: 使用LLM来评估文本，即LLM作为法官，越来越多地被大规模使用来增强甚至取代人类注释。因此，我们必须了解这样做的潜在偏见和风险。在这项工作中，我们提出了一种从法学硕士担任法官中提取高级基于概念的全球政策的方法。我们的方法由两种算法组成：1）CLoVE（对比本地可验证解释），它生成可验证的、基于概念的、对比本地解释; 2）GloVE（全球可验证解释），它使用迭代集群、总结和验证将本地规则浓缩为全球策略。我们在七个标准基准测试数据集上评估了GloVE，用于内容伤害检测。我们发现，提取的全球政策高度忠实于法学硕士作为法官的决定。此外，我们还评估了全球政策对文本扰动和对抗攻击的稳健性。最后，我们进行了一项用户研究，以评估用户对全球政策的理解和满意度。



## **6. Breaking the Reviewer: Assessing the Vulnerability of Large Language Models in Automated Peer Review Under Textual Adversarial Attacks**

打破评论者：评估文本对抗攻击下自动同行评审中大型语言模型的脆弱性 cs.CL

Minor correction: Fixed sign errors in the results table. The update  does not affect the main findings or conclusions

**SubmitDate**: 2025-10-09    [abs](http://arxiv.org/abs/2506.11113v3) [paper-pdf](http://arxiv.org/pdf/2506.11113v3)

**Authors**: Tzu-Ling Lin, Wei-Chih Chen, Teng-Fang Hsiao, Hou-I Liu, Ya-Hsin Yeh, Yu Kai Chan, Wen-Sheng Lien, Po-Yen Kuo, Philip S. Yu, Hong-Han Shuai

**Abstract**: Peer review is essential for maintaining academic quality, but the increasing volume of submissions places a significant burden on reviewers. Large language models (LLMs) offer potential assistance in this process, yet their susceptibility to textual adversarial attacks raises reliability concerns. This paper investigates the robustness of LLMs used as automated reviewers in the presence of such attacks. We focus on three key questions: (1) The effectiveness of LLMs in generating reviews compared to human reviewers. (2) The impact of adversarial attacks on the reliability of LLM-generated reviews. (3) Challenges and potential mitigation strategies for LLM-based review. Our evaluation reveals significant vulnerabilities, as text manipulations can distort LLM assessments. We offer a comprehensive evaluation of LLM performance in automated peer reviewing and analyze its robustness against adversarial attacks. Our findings emphasize the importance of addressing adversarial risks to ensure AI strengthens, rather than compromises, the integrity of scholarly communication.

摘要: 同行评审对于保持学术质量至关重要，但提交量的增加给评审者带来了沉重的负担。大型语言模型（LLM）在此过程中提供了潜在的帮助，但它们对文本对抗攻击的敏感性引发了可靠性问题。本文研究了在存在此类攻击的情况下用作自动审查员的LLM的稳健性。我们重点关注三个关键问题：（1）与人类评审员相比，LLM在生成评审方面的有效性。(2)对抗性攻击对LLM生成的评论的可靠性的影响。(3)LLM为基础的审查的挑战和潜在的缓解策略。我们的评估揭示了重大的漏洞，因为文本操作可能会扭曲LLM评估。我们提供了一个全面的评估LLM性能的自动同行评审，并分析其对抗攻击的鲁棒性。我们的研究结果强调了解决对抗风险的重要性，以确保人工智能加强而不是损害学术交流的完整性。



## **7. DNA-DetectLLM: Unveiling AI-Generated Text via a DNA-Inspired Mutation-Repair Paradigm**

DNA-DetectLLM：通过DNA启发的突变修复范式揭示人工智能生成的文本 cs.CL

NeurIPS 2025 Spotlight

**SubmitDate**: 2025-10-09    [abs](http://arxiv.org/abs/2509.15550v2) [paper-pdf](http://arxiv.org/pdf/2509.15550v2)

**Authors**: Xiaowei Zhu, Yubing Ren, Fang Fang, Qingfeng Tan, Shi Wang, Yanan Cao

**Abstract**: The rapid advancement of large language models (LLMs) has blurred the line between AI-generated and human-written text. This progress brings societal risks such as misinformation, authorship ambiguity, and intellectual property concerns, highlighting the urgent need for reliable AI-generated text detection methods. However, recent advances in generative language modeling have resulted in significant overlap between the feature distributions of human-written and AI-generated text, blurring classification boundaries and making accurate detection increasingly challenging. To address the above challenges, we propose a DNA-inspired perspective, leveraging a repair-based process to directly and interpretably capture the intrinsic differences between human-written and AI-generated text. Building on this perspective, we introduce DNA-DetectLLM, a zero-shot detection method for distinguishing AI-generated and human-written text. The method constructs an ideal AI-generated sequence for each input, iteratively repairs non-optimal tokens, and quantifies the cumulative repair effort as an interpretable detection signal. Empirical evaluations demonstrate that our method achieves state-of-the-art detection performance and exhibits strong robustness against various adversarial attacks and input lengths. Specifically, DNA-DetectLLM achieves relative improvements of 5.55% in AUROC and 2.08% in F1 score across multiple public benchmark datasets. Code and data are available at https://github.com/Xiaoweizhu57/DNA-DetectLLM.

摘要: 大型语言模型（LLM）的快速发展模糊了人工智能生成的文本和人类编写的文本之间的界限。这一进展带来了错误信息、作者身份模糊和知识产权问题等社会风险，凸显了对可靠的人工智能生成文本检测方法的迫切需求。然而，生成式语言建模的最新进展导致人类书写文本和人工智能生成文本的特征分布之间存在显着重叠，模糊了分类边界，并使准确检测变得越来越具有挑战性。为了解决上述挑战，我们提出了一种受DNA启发的视角，利用基于修复的流程来直接且可解释地捕捉人类书写和人工智能生成的文本之间的内在差异。基于这一观点，我们引入了DNA-DetectLLM，这是一种用于区分人工智能生成文本和人类编写文本的零镜头检测方法。该方法为每个输入构建理想的人工智能生成序列，迭代地修复非最优令牌，并将累积修复工作量化为可解释的检测信号。经验评估表明，我们的方法实现了最先进的检测性能，并对各种对抗攻击和输入长度表现出强大的鲁棒性。具体来说，在多个公共基准数据集中，DNA-DetectLLM在AUROC和F1评分上相对提高了5.55%，在F1评分上相对提高了2.08%。代码和数据可在https://github.com/Xiaoweizhu57/DNA-DetectLLM上获取。



## **8. Backdoor Vectors: a Task Arithmetic View on Backdoor Attacks and Defenses**

后门载体：后门攻击和防御的任务算术视图 cs.LG

22 pages, 13 figures, 15 tables

**SubmitDate**: 2025-10-09    [abs](http://arxiv.org/abs/2510.08016v1) [paper-pdf](http://arxiv.org/pdf/2510.08016v1)

**Authors**: Stanisław Pawlak, Jan Dubiński, Daniel Marczak, Bartłomiej Twardowski

**Abstract**: Model merging (MM) recently emerged as an effective method for combining large deep learning models. However, it poses significant security risks. Recent research shows that it is highly susceptible to backdoor attacks, which introduce a hidden trigger into a single fine-tuned model instance that allows the adversary to control the output of the final merged model at inference time. In this work, we propose a simple framework for understanding backdoor attacks by treating the attack itself as a task vector. $Backdoor\ Vector\ (BV)$ is calculated as the difference between the weights of a fine-tuned backdoored model and fine-tuned clean model. BVs reveal new insights into attacks understanding and a more effective framework to measure their similarity and transferability. Furthermore, we propose a novel method that enhances backdoor resilience through merging dubbed $Sparse\ Backdoor\ Vector\ (SBV)$ that combines multiple attacks into a single one. We identify the core vulnerability behind backdoor threats in MM: $inherent\ triggers$ that exploit adversarial weaknesses in the base model. To counter this, we propose $Injection\ BV\ Subtraction\ (IBVS)$ - an assumption-free defense against backdoors in MM. Our results show that SBVs surpass prior attacks and is the first method to leverage merging to improve backdoor effectiveness. At the same time, IBVS provides a lightweight, general defense that remains effective even when the backdoor threat is entirely unknown.

摘要: 模型合并（MM）最近成为一种有效的方法，用于组合大型深度学习模型。然而，它构成了重大的安全风险。最近的研究表明，它非常容易受到后门攻击，后门攻击将隐藏的触发器引入到单个微调的模型实例中，允许对手在推理时控制最终合并模型的输出。在这项工作中，我们提出了一个简单的框架来理解后门攻击的攻击本身作为一个任务向量。$Backdoor\ Vector\（BV）$计算为微调后门模型和微调干净模型的权重之间的差。BV揭示了对攻击理解的新见解，以及衡量其相似性和可移植性的更有效框架。此外，我们提出了一种新颖的方法，通过合并名为$Sparse\ Backdoor\ Vector\（SBV）$来增强后门弹性，该方法将多个攻击组合为单个攻击。我们在MM中识别出后门威胁背后的核心漏洞：$inherent\ triggers$利用基本模型中的对抗性弱点。为了解决这个问题，我们提出了$Injecting\ BV\Subtration\（IBVS）$ -一种针对MM中后门的无触发防御。我们的结果表明SBV超越了之前的攻击，是第一种利用合并来提高后门有效性的方法。与此同时，IBVS提供了一种轻量级的通用防御，即使后门威胁完全未知，也仍然有效。



## **9. Fewer Weights, More Problems: A Practical Attack on LLM Pruning**

更少的权重，更多的问题：对LLM修剪的实用攻击 cs.LG

**SubmitDate**: 2025-10-09    [abs](http://arxiv.org/abs/2510.07985v1) [paper-pdf](http://arxiv.org/pdf/2510.07985v1)

**Authors**: Kazuki Egashira, Robin Staab, Thibaud Gloaguen, Mark Vero, Martin Vechev

**Abstract**: Model pruning, i.e., removing a subset of model weights, has become a prominent approach to reducing the memory footprint of large language models (LLMs) during inference. Notably, popular inference engines, such as vLLM, enable users to conveniently prune downloaded models before they are deployed. While the utility and efficiency of pruning methods have improved significantly, the security implications of pruning remain underexplored. In this work, for the first time, we show that modern LLM pruning methods can be maliciously exploited. In particular, an adversary can construct a model that appears benign yet, once pruned, exhibits malicious behaviors. Our method is based on the idea that the adversary can compute a proxy metric that estimates how likely each parameter is to be pruned. With this information, the adversary can first inject a malicious behavior into those parameters that are unlikely to be pruned. Then, they can repair the model by using parameters that are likely to be pruned, effectively canceling out the injected behavior in the unpruned model. We demonstrate the severity of our attack through extensive evaluation on five models; after any of the pruning in vLLM are applied (Magnitude, Wanda, and SparseGPT), it consistently exhibits strong malicious behaviors in a diverse set of attack scenarios (success rates of up to $95.7\%$ for jailbreak, $98.7\%$ for benign instruction refusal, and $99.5\%$ for targeted content injection). Our results reveal a critical deployment-time security gap and underscore the urgent need for stronger security awareness in model compression.

摘要: 模型修剪，即删除模型权重的子集已成为减少推理期间大型语言模型（LLM）内存占用的一种主要方法。值得注意的是，vLLM等流行推理引擎使用户能够在部署下载的模型之前方便地修剪它们。虽然修剪方法的实用性和效率有了显着提高，但修剪的安全影响仍然没有得到充分的研究。在这项工作中，我们首次表明现代LLM修剪方法可以被恶意利用。特别是，对手可以构建一个看起来良性但一旦修剪，就会表现出恶意行为的模型。我们的方法基于这样的想法：对手可以计算代理指标，该指标估计每个参数被修剪的可能性。有了这些信息，对手可以首先将恶意行为注入到那些不太可能被修剪的参数中。然后，他们可以通过使用可能被修剪的参数来修复模型，从而有效地抵消未修剪模型中注入的行为。我们通过对五个模型的广泛评估来证明攻击的严重性;应用vLLM中的任何修剪（Magnitude、Wanda和SparseGPT）后，它在各种攻击场景中始终表现出强烈的恶意行为（越狱成功率高达95.7美元，良性指令拒绝成功率高达98.7美元，定向内容注入成功率高达99.5美元）。我们的结果揭示了一个关键的部署时安全差距，并强调了模型压缩中迫切需要更强的安全意识。



## **10. Safe-Control: A Safety Patch for Mitigating Unsafe Content in Text-to-Image Generation Models**

Safe-Control：用于缓解文本到图像生成模型中不安全内容的安全补丁 cs.CV

**SubmitDate**: 2025-10-09    [abs](http://arxiv.org/abs/2508.21099v2) [paper-pdf](http://arxiv.org/pdf/2508.21099v2)

**Authors**: Xiangtao Meng, Yingkai Dong, Ning Yu, Li Wang, Zheng Li, Shanqing Guo

**Abstract**: Despite the advancements in Text-to-Image (T2I) generation models, their potential for misuse or even abuse raises serious safety concerns. Model developers have made tremendous efforts to introduce safety mechanisms that can address these concerns in T2I models. However, the existing safety mechanisms, whether external or internal, either remain susceptible to evasion under distribution shifts or require extensive model-specific adjustments. To address these limitations, we introduce Safe-Control, an innovative plug-and-play safety patch designed to mitigate unsafe content generation in T2I models. Using data-driven strategies and safety-aware conditions, Safe-Control injects safety control signals into the locked T2I model, acting as an update in a patch-like manner. Model developers can also construct various safety patches to meet the evolving safety requirements, which can be flexibly merged into a single, unified patch. Its plug-and-play design further ensures adaptability, making it compatible with other T2I models of similar denoising architecture. We conduct extensive evaluations on six diverse and public T2I models. Empirical results highlight that Safe-Control is effective in reducing unsafe content generation across six diverse T2I models with similar generative architectures, yet it successfully maintains the quality and text alignment of benign images. Compared to seven state-of-the-art safety mechanisms, including both external and internal defenses, Safe-Control significantly outperforms all baselines in reducing unsafe content generation. For example, it reduces the probability of unsafe content generation to 7%, compared to approximately 20% for most baseline methods, under both unsafe prompts and the latest adversarial attacks.

摘要: 尽管文本到图像（T2 I）生成模型取得了进步，但它们被滥用甚至滥用的可能性引发了严重的安全问题。模型开发人员做出了巨大努力来引入可以解决T2 I模型中这些问题的安全机制。然而，现有的安全机制，无论是外部的还是内部的，要么仍然容易在分配转移下被规避，要么需要针对特定模型的广泛调整。为了解决这些限制，我们引入了Safe-Control，这是一种创新的即插即用安全补丁，旨在减轻T2 I模型中的不安全内容生成。Safe-Control使用数据驱动策略和安全意识条件，将安全控制信号注入锁定的T2 I模型，以类似补丁的方式充当更新。模型开发人员还可以构建各种安全补丁来满足不断变化的安全要求，这些补丁可以灵活地合并到单个、统一的补丁中。其即插即用设计进一步确保了适应性，使其与类似去噪架构的其他T2 I型号兼容。我们对六种多样化的公共T2 I模型进行了广泛的评估。经验结果强调，Safe-Control可以有效减少具有相似生成架构的六种不同T2 I模型中的不安全内容生成，但它成功地保持了良性图像的质量和文本对齐。与七种最先进的安全机制（包括外部和内部防御）相比，Safe-Control在减少不安全内容生成方面显着优于所有基线。例如，在不安全提示和最新的对抗性攻击下，它将不安全内容生成的可能性降低到7%，而大多数基线方法的可能性约为20%。



## **11. Bloodroot: When Watermarking Turns Poisonous For Stealthy Backdoor**

Bloodroot：当水印对隐形后门有毒时 eess.AS

5 pages, 3 figures

**SubmitDate**: 2025-10-09    [abs](http://arxiv.org/abs/2510.07909v1) [paper-pdf](http://arxiv.org/pdf/2510.07909v1)

**Authors**: Kuan-Yu Chen, Yi-Cheng Lin, Jeng-Lin Li, Jian-Jiun Ding

**Abstract**: Backdoor data poisoning is a crucial technique for ownership protection and defending against malicious attacks. Embedding hidden triggers in training data can manipulate model outputs, enabling provenance verification, and deterring unauthorized use. However, current audio backdoor methods are suboptimal, as poisoned audio often exhibits degraded perceptual quality, which is noticeable to human listeners. This work explores the intrinsic stealthiness and effectiveness of audio watermarking in achieving successful poisoning. We propose a novel Watermark-as-Trigger concept, integrated into the Bloodroot backdoor framework via adversarial LoRA fine-tuning, which enhances perceptual quality while achieving a much higher trigger success rate and clean-sample accuracy. Experiments on speech recognition (SR) and speaker identification (SID) datasets show that watermark-based poisoning remains effective under acoustic filtering and model pruning. The proposed Bloodroot backdoor framework not only secures data-to-model ownership, but also well reveals the risk of adversarial misuse.

摘要: 后门数据中毒是保护所有权和防御恶意攻击的关键技术。在训练数据中嵌入隐藏触发器可以操纵模型输出，实现出处验证并阻止未经授权的使用。然而，当前的音频后门方法并不是最佳的，因为中毒音频通常表现出下降的感知质量，这对人类听众来说是显而易见的。这项工作探讨了音频水印在实现成功中毒方面的内在隐秘性和有效性。我们提出了一种新颖的水印即触发器概念，通过对抗性LoRA微调集成到Bloodroot后门框架中，从而增强了感知质量，同时实现了更高的触发成功率和干净样本准确性。语音识别（SR）和说话人识别（IDS）数据集的实验表明，基于水印的中毒在声学过滤和模型修剪下仍然有效。拟议的Bloodroot后门框架不仅确保了数据到模型的所有权，而且还很好地揭示了对抗性滥用的风险。



## **12. AEGIS : Automated Co-Evolutionary Framework for Guarding Prompt Injections Schema**

AEGIS：守卫提示注射模式的自动协同进化框架 cs.CR

**SubmitDate**: 2025-10-09    [abs](http://arxiv.org/abs/2509.00088v2) [paper-pdf](http://arxiv.org/pdf/2509.00088v2)

**Authors**: Ting-Chun Liu, Ching-Yu Hsu, Kuan-Yi Lee, Chi-An Fu, Hung-yi Lee

**Abstract**: Prompt injection attacks pose a significant challenge to the safe deployment of Large Language Models (LLMs) in real-world applications. While prompt-based detection offers a lightweight and interpretable defense strategy, its effectiveness has been hindered by the need for manual prompt engineering. To address this issue, we propose AEGIS , an Automated co-Evolutionary framework for Guarding prompt Injections Schema. Both attack and defense prompts are iteratively optimized against each other using a gradient-like natural language prompt optimization technique. This framework enables both attackers and defenders to autonomously evolve via a Textual Gradient Optimization (TGO) module, leveraging feedback from an LLM-guided evaluation loop. We evaluate our system on a real-world assignment grading dataset of prompt injection attacks and demonstrate that our method consistently outperforms existing baselines, achieving superior robustness in both attack success and detection. Specifically, the attack success rate (ASR) reaches 1.0, representing an improvement of 0.26 over the baseline. For detection, the true positive rate (TPR) improves by 0.23 compared to the previous best work, reaching 0.84, and the true negative rate (TNR) remains comparable at 0.89. Ablation studies confirm the importance of co-evolution, gradient buffering, and multi-objective optimization. We also confirm that this framework is effective in different LLMs. Our results highlight the promise of adversarial training as a scalable and effective approach for guarding prompt injections.

摘要: 提示注入攻击对现实世界应用程序中大型语言模型（LLM）的安全部署构成了重大挑战。虽然基于预算的检测提供了一种轻量级且可解释的防御策略，但其有效性因需要手动提示工程而受到阻碍。为了解决这个问题，我们提出了AEGIS，这是Guarding提示注射模式的自动协同进化框架。攻击和防御提示都使用类似梯度的自然语言提示优化技术进行相互迭代优化。该框架使攻击者和防御者能够通过文本梯度优化（TGO）模块自主进化，利用来自LLM指导评估循环的反馈。我们在即时注入攻击的现实世界分配分级数据集上评估了我们的系统，并证明我们的方法始终优于现有基线，在攻击成功和检测方面都实现了卓越的鲁棒性。具体来说，攻击成功率（ASB）达到1.0，比基线提高0.26。在检测方面，真阳性率（TLR）与之前的最佳工作相比提高了0.23，达到0.84，真阴性率（TNR）保持在0.89相当。消融研究证实了共同进化、梯度缓冲和多目标优化的重要性。我们还确认该框架在不同的LLM中有效。我们的结果凸显了对抗训练作为一种可扩展且有效的预防及时注射方法的前景。



## **13. Rethinking Reasoning: A Survey on Reasoning-based Backdoors in LLMs**

重新思考推理：LLM中基于推理的后门调查 cs.CR

**SubmitDate**: 2025-10-09    [abs](http://arxiv.org/abs/2510.07697v1) [paper-pdf](http://arxiv.org/pdf/2510.07697v1)

**Authors**: Man Hu, Xinyi Wu, Zuofeng Suo, Jinbo Feng, Linghui Meng, Yanhao Jia, Anh Tuan Luu, Shuai Zhao

**Abstract**: With the rise of advanced reasoning capabilities, large language models (LLMs) are receiving increasing attention. However, although reasoning improves LLMs' performance on downstream tasks, it also introduces new security risks, as adversaries can exploit these capabilities to conduct backdoor attacks. Existing surveys on backdoor attacks and reasoning security offer comprehensive overviews but lack in-depth analysis of backdoor attacks and defenses targeting LLMs' reasoning abilities. In this paper, we take the first step toward providing a comprehensive review of reasoning-based backdoor attacks in LLMs by analyzing their underlying mechanisms, methodological frameworks, and unresolved challenges. Specifically, we introduce a new taxonomy that offers a unified perspective for summarizing existing approaches, categorizing reasoning-based backdoor attacks into associative, passive, and active. We also present defense strategies against such attacks and discuss current challenges alongside potential directions for future research. This work offers a novel perspective, paving the way for further exploration of secure and trustworthy LLM communities.

摘要: 随着高级推理能力的兴起，大型语言模型（LLM）越来越受到关注。然而，尽管推理提高了LLM在下游任务上的性能，但它也带来了新的安全风险，因为对手可以利用这些能力进行后门攻击。现有的关于后门攻击和推理安全性的调查提供了全面的概述，但缺乏对针对LLM推理能力的后门攻击和防御的深入分析。在本文中，我们通过分析LLM中基于推理的后门攻击的潜在机制、方法框架和未解决的挑战，迈出了对LLM中基于推理的后门攻击进行全面审查的第一步。具体来说，我们引入了一个新的分类法，提供了一个统一的角度来总结现有的方法，分类基于推理的后门攻击为关联，被动和主动。我们还提出了针对此类攻击的防御策略，并讨论了当前的挑战以及未来研究的潜在方向。这项工作提供了一个新的视角，为进一步探索安全和值得信赖的LLM社区铺平了道路。



## **14. DGTEN: A Robust Deep Gaussian based Graph Neural Network for Dynamic Trust Evaluation with Uncertainty-Quantification Support**

DGTON：一个鲁棒的基于深度高斯的图神经网络，用于动态信任评估，并支持不确定性量化 cs.LG

18 pages, 9 figures, 5 tables

**SubmitDate**: 2025-10-08    [abs](http://arxiv.org/abs/2510.07620v1) [paper-pdf](http://arxiv.org/pdf/2510.07620v1)

**Authors**: Muhammad Usman, Yugyung Lee

**Abstract**: Dynamic trust evaluation in large, rapidly evolving graphs requires models that can capture changing relationships, express calibrated confidence, and resist adversarial manipulation. DGTEN (Deep Gaussian-based Trust Evaluation Network) introduces a unified graph framework that achieves all three by combining uncertainty-aware message passing, expressive temporal modeling, and built-in defenses against trust-targeted attacks. It represents nodes and edges as Gaussian distributions so that both semantic signals and epistemic uncertainty propagate through the graph neural network, enabling risk-aware trust decisions rather than overconfident guesses. To model how trust evolves, it employs hybrid Absolute-Gaussian-Hourglass (HAGH) positional encoding with Kolmogorov-Arnold network-based unbiased multi-head attention, followed by an ordinary differential equation (ODE)-based residual learning module to jointly capture abrupt shifts and smooth trends. Robust adaptive ensemble coefficient analysis prunes or down-weights suspicious interactions using complementary cosine and Jaccard similarity measures, mitigating reputation laundering, sabotage, and on/off attacks. On two signed Bitcoin trust networks, DGTEN delivers significant improvements: in single-timeslot prediction on Bitcoin-Alpha, it improves MCC by 10.77% over the best dynamic baseline; in the cold-start scenario, it achieves a 16.41% MCC gain - the largest across all tasks and datasets. Under adversarial on/off attacks, it surpasses the baseline by up to 11.63% MCC. These results validate the effectiveness of the unified DGTEN framework.

摘要: 在快速发展的大型图表中进行动态信任评估需要能够捕捉不断变化的关系、表达校准的信心并抵抗对抗操纵的模型。DGTON（基于深度高斯的信任评估网络）引入了一个统一的图框架，该框架通过结合不确定性感知消息传递、表达性时态建模和针对以信任为目标的攻击的内置防御来实现这三者。它将节点和边表示为高斯分布，以便语义信号和认识不确定性都通过图神经网络传播，从而实现风险感知的信任决策，而不是过度自信的猜测。为了对信任如何演变进行建模，它采用混合绝对高斯沙漏（HAGH）位置编码和基于Kolmogorov-Arnold网络的无偏多头注意力，然后采用基于常微方程（ODE）的剩余学习模块来联合捕捉突然的变化和平滑的趋势。稳健的自适应集成系数分析使用互补的Cosin和Jaccard相似性测量来修剪或淡化可疑的交互，从而减轻声誉洗钱、破坏和开/关攻击。在两个已签署的比特币信任网络上，DGTON提供了显着改进：在Bitcoin-Alpha的单时段预测中，它将MCC比最佳动态基线提高了10.77%;在冷启动场景中，它实现了16.41%的MCC收益--所有任务和数据集中最大的。在对抗性开/关攻击下，其超出基线高达11.63% MCC。这些结果验证了统一DGTON框架的有效性。



## **15. $\textit{Agents Under Siege}$: Breaking Pragmatic Multi-Agent LLM Systems with Optimized Prompt Attacks**

$\texttit {Agents Under Siege}$：通过优化的即时攻击破解实用多Agent LLM系统 cs.MA

**SubmitDate**: 2025-10-08    [abs](http://arxiv.org/abs/2504.00218v2) [paper-pdf](http://arxiv.org/pdf/2504.00218v2)

**Authors**: Rana Muhammad Shahroz Khan, Zhen Tan, Sukwon Yun, Charles Fleming, Tianlong Chen

**Abstract**: Most discussions about Large Language Model (LLM) safety have focused on single-agent settings but multi-agent LLM systems now create novel adversarial risks because their behavior depends on communication between agents and decentralized reasoning. In this work, we innovatively focus on attacking pragmatic systems that have constrains such as limited token bandwidth, latency between message delivery, and defense mechanisms. We design a $\textit{permutation-invariant adversarial attack}$ that optimizes prompt distribution across latency and bandwidth-constraint network topologies to bypass distributed safety mechanisms within the system. Formulating the attack path as a problem of $\textit{maximum-flow minimum-cost}$, coupled with the novel $\textit{Permutation-Invariant Evasion Loss (PIEL)}$, we leverage graph-based optimization to maximize attack success rate while minimizing detection risk. Evaluating across models including $\texttt{Llama}$, $\texttt{Mistral}$, $\texttt{Gemma}$, $\texttt{DeepSeek}$ and other variants on various datasets like $\texttt{JailBreakBench}$ and $\texttt{AdversarialBench}$, our method outperforms conventional attacks by up to $7\times$, exposing critical vulnerabilities in multi-agent systems. Moreover, we demonstrate that existing defenses, including variants of $\texttt{Llama-Guard}$ and $\texttt{PromptGuard}$, fail to prohibit our attack, emphasizing the urgent need for multi-agent specific safety mechanisms.

摘要: Most discussions about Large Language Model (LLM) safety have focused on single-agent settings but multi-agent LLM systems now create novel adversarial risks because their behavior depends on communication between agents and decentralized reasoning. In this work, we innovatively focus on attacking pragmatic systems that have constrains such as limited token bandwidth, latency between message delivery, and defense mechanisms. We design a $\textit{permutation-invariant adversarial attack}$ that optimizes prompt distribution across latency and bandwidth-constraint network topologies to bypass distributed safety mechanisms within the system. Formulating the attack path as a problem of $\textit{maximum-flow minimum-cost}$, coupled with the novel $\textit{Permutation-Invariant Evasion Loss (PIEL)}$, we leverage graph-based optimization to maximize attack success rate while minimizing detection risk. Evaluating across models including $\texttt{Llama}$, $\texttt{Mistral}$, $\texttt{Gemma}$, $\texttt{DeepSeek}$ and other variants on various datasets like $\texttt{JailBreakBench}$ and $\texttt{AdversarialBench}$, our method outperforms conventional attacks by up to $7\times$, exposing critical vulnerabilities in multi-agent systems. Moreover, we demonstrate that existing defenses, including variants of $\texttt{Llama-Guard}$ and $\texttt{PromptGuard}$, fail to prohibit our attack, emphasizing the urgent need for multi-agent specific safety mechanisms.



## **16. MeanSparse: Post-Training Robustness Enhancement Through Mean-Centered Feature Sparsification**

MeanSparse：通过以均值为中心的特征稀疏化来增强训练后的鲁棒性 cs.CV

**SubmitDate**: 2025-10-08    [abs](http://arxiv.org/abs/2406.05927v3) [paper-pdf](http://arxiv.org/pdf/2406.05927v3)

**Authors**: Sajjad Amini, Mohammadreza Teymoorianfard, Shiqing Ma, Amir Houmansadr

**Abstract**: We present a simple yet effective method to improve the robustness of both Convolutional and attention-based Neural Networks against adversarial examples by post-processing an adversarially trained model. Our technique, MeanSparse, cascades the activation functions of a trained model with novel operators that sparsify mean-centered feature vectors. This is equivalent to reducing feature variations around the mean, and we show that such reduced variations merely affect the model's utility, yet they strongly attenuate the adversarial perturbations and decrease the attacker's success rate. Our experiments show that, when applied to the top models in the RobustBench leaderboard, MeanSparse achieves a new robustness record of 75.28% (from 73.71%), 44.78% (from 42.67%) and 62.12% (from 59.56%) on CIFAR-10, CIFAR-100 and ImageNet, respectively, in terms of AutoAttack accuracy. Code is available at https://github.com/SPIN-UMass/MeanSparse

摘要: 我们提出了一种简单而有效的方法，通过对对抗训练模型进行后处理来提高卷积神经网络和基于注意力的神经网络对对抗样本的鲁棒性。我们的技术MeanSparse将训练模型的激活函数与新的操作符级联，这些操作符使以平均值为中心的特征向量稀疏化。这相当于减少平均值周围的特征变化，我们表明，这种减少的变化只会影响模型的效用，但它们强烈减弱了对抗性扰动，降低了攻击者的成功率。我们的实验表明，当应用于RobustBench排行榜中的顶级模型时，MeanSparse在CIFAR-10，CIFAR-100和ImageNet上分别实现了75.28%（从73.71%），44.78%（从42.67%）和62.12%（从59.56%）的新鲁棒性记录。代码可访问https://github.com/SPIN-UMass/MeanSparse



## **17. LLMs Encode Harmfulness and Refusal Separately**

法学硕士分别对有害和拒绝进行编码 cs.CL

**SubmitDate**: 2025-10-08    [abs](http://arxiv.org/abs/2507.11878v3) [paper-pdf](http://arxiv.org/pdf/2507.11878v3)

**Authors**: Jiachen Zhao, Jing Huang, Zhengxuan Wu, David Bau, Weiyan Shi

**Abstract**: LLMs are trained to refuse harmful instructions, but do they truly understand harmfulness beyond just refusing? Prior work has shown that LLMs' refusal behaviors can be mediated by a one-dimensional subspace, i.e., a refusal direction. In this work, we identify a new dimension to analyze safety mechanisms in LLMs, i.e., harmfulness, which is encoded internally as a separate concept from refusal. There exists a harmfulness direction that is distinct from the refusal direction. As causal evidence, steering along the harmfulness direction can lead LLMs to interpret harmless instructions as harmful, but steering along the refusal direction tends to elicit refusal responses directly without reversing the model's judgment on harmfulness. Furthermore, using our identified harmfulness concept, we find that certain jailbreak methods work by reducing the refusal signals without reversing the model's internal belief of harmfulness. We also find that adversarially finetuning models to accept harmful instructions has minimal impact on the model's internal belief of harmfulness. These insights lead to a practical safety application: The model's latent harmfulness representation can serve as an intrinsic safeguard (Latent Guard) for detecting unsafe inputs and reducing over-refusals that is robust to finetuning attacks. For instance, our Latent Guard achieves performance comparable to or better than Llama Guard 3 8B, a dedicated finetuned safeguard model, across different jailbreak methods. Our findings suggest that LLMs' internal understanding of harmfulness is more robust than their refusal decision to diverse input instructions, offering a new perspective to study AI safety.

摘要: 法学硕士被训练去拒绝有害的指令，但是他们真的理解除了拒绝之外的危害吗？先前的工作已经表明，LLM的拒绝行为可以由一维子空间介导，即，拒绝的方向。在这项工作中，我们确定了一个新的维度来分析LLM中的安全机制，即，危害性，它在内部被编码为与拒绝分开的概念。存在一个与拒绝方向不同的危害方向。作为因果证据，沿着有害方向引导可能会导致LLM将无害的指令解释为有害的，但沿着拒绝方向引导往往会直接引发拒绝反应，而不会扭转模型对有害性的判断。此外，使用我们确定的危害性概念，我们发现某些越狱方法通过减少拒绝信号来发挥作用，而不会扭转模型内部的危害性信念。我们还发现，对模型进行不利调整以接受有害指令对模型内部有害性信念的影响最小。这些见解导致了实际的安全应用：该模型的潜在危害性表示可以作为内在保护措施（潜在保护措施），用于检测不安全的输入并减少过度拒绝，这对微调攻击具有鲁棒性。例如，我们的潜伏卫士在不同的越狱方法中实现了与Llama Guard 3 8 B相当或更好的性能，Llama Guard 3 8 B是一种专用的微调保护模型。我们的研究结果表明，LLM对危害性的内部理解比他们拒绝多种输入指令的决定更强大，为研究人工智能安全性提供了新的视角。



## **18. D2RA: Dual Domain Regeneration Attack**

D2 RA：双域再生攻击 cs.CV

**SubmitDate**: 2025-10-08    [abs](http://arxiv.org/abs/2510.07538v1) [paper-pdf](http://arxiv.org/pdf/2510.07538v1)

**Authors**: Pragati Shuddhodhan Meshram, Varun Chandrasekaran

**Abstract**: The growing use of generative models has intensified the need for watermarking methods that ensure content attribution and provenance. While recent semantic watermarking schemes improve robustness by embedding signals in latent or frequency representations, we show they remain vulnerable even under resource-constrained adversarial settings. We present D2RA, a training-free, single-image attack that removes or weakens watermarks without access to the underlying model. By projecting watermarked images onto natural priors across complementary representations, D2RA suppresses watermark signals while preserving visual fidelity. Experiments across diverse watermarking schemes demonstrate that our approach consistently reduces watermark detectability, revealing fundamental weaknesses in current designs. Our code is available at https://github.com/Pragati-Meshram/DAWN.

摘要: 生成模型的越来越多的使用加剧了对确保内容属性和出处的水印方法的需求。虽然最近的语义水印方案通过将信号嵌入潜在或频率表示来提高鲁棒性，但我们表明，即使在资源有限的对抗环境下，它们仍然很脆弱。我们提出了D2 RA，这是一种无需训练的单图像攻击，可以在不访问底层模型的情况下删除或削弱水印。通过将带水印的图像投影到互补表示的自然先验上，D2 RA抑制了水印信号，同时保留了视觉保真度。各种水印方案的实验表明，我们的方法始终降低了水印的可检测性，揭示了当前设计中的根本弱点。我们的代码可在https://github.com/Pragati-Meshram/DAWN上获取。



## **19. PEAR: Planner-Executor Agent Robustness Benchmark**

PEAR：规划者-执行者代理稳健性基准 cs.LG

**SubmitDate**: 2025-10-08    [abs](http://arxiv.org/abs/2510.07505v1) [paper-pdf](http://arxiv.org/pdf/2510.07505v1)

**Authors**: Shen Dong, Mingxuan Zhang, Pengfei He, Li Ma, Bhavani Thuraisingham, Hui Liu, Yue Xing

**Abstract**: Large Language Model (LLM)-based Multi-Agent Systems (MAS) have emerged as a powerful paradigm for tackling complex, multi-step tasks across diverse domains. However, despite their impressive capabilities, MAS remain susceptible to adversarial manipulation. Existing studies typically examine isolated attack surfaces or specific scenarios, leaving a lack of holistic understanding of MAS vulnerabilities. To bridge this gap, we introduce PEAR, a benchmark for systematically evaluating both the utility and vulnerability of planner-executor MAS. While compatible with various MAS architectures, our benchmark focuses on the planner-executor structure, which is a practical and widely adopted design. Through extensive experiments, we find that (1) a weak planner degrades overall clean task performance more severely than a weak executor; (2) while a memory module is essential for the planner, having a memory module for the executor does not impact the clean task performance; (3) there exists a trade-off between task performance and robustness; and (4) attacks targeting the planner are particularly effective at misleading the system. These findings offer actionable insights for enhancing the robustness of MAS and lay the groundwork for principled defenses in multi-agent settings.

摘要: 基于大型语言模型（LLM）的多智能体系统（MAS）已成为处理跨不同领域复杂、多步骤任务的强大范式。然而，尽管MAS的能力令人印象深刻，但仍然容易受到对抗操纵。现有的研究通常会检查孤立的攻击表面或特定场景，从而缺乏对MAS漏洞的全面了解。为了弥合这一差距，我们引入了PEAR，这是一个用于系统评估规划者-执行者MAS的实用性和脆弱性的基准。虽然兼容各种MAS体系结构，我们的基准集中在规划者-执行器结构，这是一个实用的和广泛采用的设计。通过大量的实验，我们发现：（1）弱规划器比弱执行器更严重地降低了清洁任务的整体性能;（2）虽然规划器的内存模块是必不可少的，但执行器的内存模块并不影响清洁任务的性能;（3）任务性能和鲁棒性之间存在权衡;以及（4）针对计划者的攻击在误导系统方面特别有效。这些发现提供了可操作的见解，提高MAS的鲁棒性，并奠定了基础，在多智能体设置的原则性防御。



## **20. SpecGuard: Spectral Projection-based Advanced Invisible Watermarking**

SpecGuard：基于光谱投影的高级隐形水印 cs.CV

ICCV 2025 Accepted Paper

**SubmitDate**: 2025-10-08    [abs](http://arxiv.org/abs/2510.07302v1) [paper-pdf](http://arxiv.org/pdf/2510.07302v1)

**Authors**: Inzamamul Alam, Md Tanvir Islam, Khan Muhammad, Simon S. Woo

**Abstract**: Watermarking embeds imperceptible patterns into images for authenticity verification. However, existing methods often lack robustness against various transformations primarily including distortions, image regeneration, and adversarial perturbation, creating real-world challenges. In this work, we introduce SpecGuard, a novel watermarking approach for robust and invisible image watermarking. Unlike prior approaches, we embed the message inside hidden convolution layers by converting from the spatial domain to the frequency domain using spectral projection of a higher frequency band that is decomposed by wavelet projection. Spectral projection employs Fast Fourier Transform approximation to transform spatial data into the frequency domain efficiently. In the encoding phase, a strength factor enhances resilience against diverse attacks, including adversarial, geometric, and regeneration-based distortions, ensuring the preservation of copyrighted information. Meanwhile, the decoder leverages Parseval's theorem to effectively learn and extract the watermark pattern, enabling accurate retrieval under challenging transformations. We evaluate the proposed SpecGuard based on the embedded watermark's invisibility, capacity, and robustness. Comprehensive experiments demonstrate the proposed SpecGuard outperforms the state-of-the-art models. To ensure reproducibility, the full code is released on \href{https://github.com/inzamamulDU/SpecGuard_ICCV_2025}{\textcolor{blue}{\textbf{GitHub}}}.

摘要: 水印将难以感知的模式嵌入图像中以进行真实性验证。然而，现有的方法通常缺乏对各种变换（主要包括失真、图像再生和对抗性扰动）的鲁棒性，从而带来现实世界的挑战。在这项工作中，我们引入了SpecGuard，这是一种用于鲁棒且不可见图像水印的新型水印方法。与以前的方法不同，我们通过使用由子波投影分解的较高频段的频谱投影从空间域转换到频域，将消息嵌入隐藏卷积层中。光谱投影采用快速傅里叶变换逼近将空间数据有效地转换到频域。在编码阶段，强度因子可以增强针对各种攻击的弹性，包括对抗性、几何性和基于再生的扭曲，确保保护受版权保护的信息。与此同时，解码器利用帕西瓦尔定理有效地学习和提取水印模式，从而在具有挑战性的转换下实现准确检索。我们根据嵌入水印的不可见性、容量和鲁棒性来评估拟议的SpecGuard。全面的实验表明，拟议的SpecGuard优于最先进的模型。为了确保可重复性，完整代码发布在\href{https：//github.com/inzamamulDU/SpecGuard_ICCV_2025}{\textColor{blue}{\textBF{GitHub}}上。



## **21. L2M-AID: Autonomous Cyber-Physical Defense by Fusing Semantic Reasoning of Large Language Models with Multi-Agent Reinforcement Learning (Preprint)**

L2 M-AID：通过融合大型语言模型的语义推理与多智能体强化学习来自主网络物理防御（预印本） cs.AI

This preprint was submitted to IEEE TrustCom 2025. The accepted  version will be published under copyright 2025 IEEE

**SubmitDate**: 2025-10-08    [abs](http://arxiv.org/abs/2510.07363v1) [paper-pdf](http://arxiv.org/pdf/2510.07363v1)

**Authors**: Tianxiang Xu, Zhichao Wen, Xinyu Zhao, Jun Wang, Yan Li, Chang Liu

**Abstract**: The increasing integration of Industrial IoT (IIoT) exposes critical cyber-physical systems to sophisticated, multi-stage attacks that elude traditional defenses lacking contextual awareness. This paper introduces L2M-AID, a novel framework for Autonomous Industrial Defense using LLM-empowered, Multi-agent reinforcement learning. L2M-AID orchestrates a team of collaborative agents, each driven by a Large Language Model (LLM), to achieve adaptive and resilient security. The core innovation lies in the deep fusion of two AI paradigms: we leverage an LLM as a semantic bridge to translate vast, unstructured telemetry into a rich, contextual state representation, enabling agents to reason about adversary intent rather than merely matching patterns. This semantically-aware state empowers a Multi-Agent Reinforcement Learning (MARL) algorithm, MAPPO, to learn complex cooperative strategies. The MARL reward function is uniquely engineered to balance security objectives (threat neutralization) with operational imperatives, explicitly penalizing actions that disrupt physical process stability. To validate our approach, we conduct extensive experiments on the benchmark SWaT dataset and a novel synthetic dataset generated based on the MITRE ATT&CK for ICS framework. Results demonstrate that L2M-AID significantly outperforms traditional IDS, deep learning anomaly detectors, and single-agent RL baselines across key metrics, achieving a 97.2% detection rate while reducing false positives by over 80% and improving response times by a factor of four. Crucially, it demonstrates superior performance in maintaining physical process stability, presenting a robust new paradigm for securing critical national infrastructure.

摘要: 工业物联网（IIoT）的日益集成使关键的网络物理系统面临复杂的多阶段攻击，这些攻击无法逃避缺乏上下文感知的传统防御。本文介绍了L2 M-AID，这是一种新型的自主工业防御框架，使用LLM授权的多智能体强化学习。L2 M-AID组织了一个协作代理团队，每个代理都由大型语言模型（LLM）驱动，以实现自适应和弹性的安全性。核心创新在于两种人工智能范式的深度融合：我们利用LLM作为语义桥梁，将庞大的非结构化遥感数据转化为丰富的上下文状态表示，使代理能够推理对手意图，而不仅仅是匹配模式。这种语义感知状态使多智能体强化学习（MARL）算法MAPPO能够学习复杂的合作策略。MARL奖励功能经过独特设计，旨在平衡安全目标（威胁消除）与运营必要性，明确惩罚破坏物理过程稳定性的行为。为了验证我们的方法，我们对基准SWaT数据集和基于MITRE ATA & CK for ICS框架生成的新型合成数据集进行了广泛的实验。结果表明，L2 M-AID在关键指标上的表现显着优于传统IDS、深度学习异常检测器和单代理RL基线，实现了97.2%的检测率，同时将误报率降低了80%以上，并将响应时间提高了四倍。至关重要的是，它在维持物理过程稳定性方面表现出色，为保护关键国家基础设施提供了强大的新范式。



## **22. Differential Privacy for Adaptive Weight Aggregation in Federated Tumor Segmentation**

联邦肿瘤分割中自适应权重聚集的差异隐私 cs.LG

I have changed the methodology because of some technical errors in  this version

**SubmitDate**: 2025-10-08    [abs](http://arxiv.org/abs/2308.00856v2) [paper-pdf](http://arxiv.org/pdf/2308.00856v2)

**Authors**: Muhammad Irfan Khan, Esa Alhoniemi, Elina Kontio, Suleiman A. Khan, Mojtaba Jafaritadi

**Abstract**: Federated Learning (FL) is a distributed machine learning approach that safeguards privacy by creating an impartial global model while respecting the privacy of individual client data. However, the conventional FL method can introduce security risks when dealing with diverse client data, potentially compromising privacy and data integrity. To address these challenges, we present a differential privacy (DP) federated deep learning framework in medical image segmentation. In this paper, we extend our similarity weight aggregation (SimAgg) method to DP-SimAgg algorithm, a differentially private similarity-weighted aggregation algorithm for brain tumor segmentation in multi-modal magnetic resonance imaging (MRI). Our DP-SimAgg method not only enhances model segmentation capabilities but also provides an additional layer of privacy preservation. Extensive benchmarking and evaluation of our framework, with computational performance as a key consideration, demonstrate that DP-SimAgg enables accurate and robust brain tumor segmentation while minimizing communication costs during model training. This advancement is crucial for preserving the privacy of medical image data and safeguarding sensitive information. In conclusion, adding a differential privacy layer in the global weight aggregation phase of the federated brain tumor segmentation provides a promising solution to privacy concerns without compromising segmentation model efficacy. By leveraging DP, we ensure the protection of client data against adversarial attacks and malicious participants.

摘要: 联合学习（FL）是一种分布式机器学习方法，通过创建公正的全球模型同时尊重个人客户数据的隐私来保护隐私。然而，传统的FL方法在处理不同的客户端数据时可能会带来安全风险，从而可能损害隐私和数据完整性。为了应对这些挑战，我们在医学图像分割中提出了一种差异隐私（DP）联邦深度学习框架。在本文中，我们将我们的相似性加权聚集（SimAgg）方法扩展到DP-SimAgg算法，这是一种用于多模式磁共振成像（MRI）中脑肿瘤分割的差异私有相似性加权聚集算法。我们的DP-SimAgg方法不仅增强了模型分割能力，还提供了额外的隐私保护层。以计算性能为关键考虑因素，对我们的框架进行了广泛的基准测试和评估，证明DP-SimAgg能够实现准确、稳健的脑肿瘤分割，同时最大限度地降低模型训练期间的通信成本。这一进步对于保护医学图像数据的隐私和保护敏感信息至关重要。总之，在联邦脑肿瘤分割的全局权重聚集阶段添加差异隐私层为隐私问题提供了一种有希望的解决方案，而不会损害分割模型的功效。通过利用DP，我们确保客户数据免受对抗攻击和恶意参与者的侵害。



## **23. Poisoning Attacks on LLMs Require a Near-constant Number of Poison Samples**

对LLM的中毒攻击需要几乎恒定数量的毒物样本 cs.LG

**SubmitDate**: 2025-10-08    [abs](http://arxiv.org/abs/2510.07192v1) [paper-pdf](http://arxiv.org/pdf/2510.07192v1)

**Authors**: Alexandra Souly, Javier Rando, Ed Chapman, Xander Davies, Burak Hasircioglu, Ezzeldin Shereen, Carlos Mougan, Vasilios Mavroudis, Erik Jones, Chris Hicks, Nicholas Carlini, Yarin Gal, Robert Kirk

**Abstract**: Poisoning attacks can compromise the safety of large language models (LLMs) by injecting malicious documents into their training data. Existing work has studied pretraining poisoning assuming adversaries control a percentage of the training corpus. However, for large models, even small percentages translate to impractically large amounts of data. This work demonstrates for the first time that poisoning attacks instead require a near-constant number of documents regardless of dataset size. We conduct the largest pretraining poisoning experiments to date, pretraining models from 600M to 13B parameters on chinchilla-optimal datasets (6B to 260B tokens). We find that 250 poisoned documents similarly compromise models across all model and dataset sizes, despite the largest models training on more than 20 times more clean data. We also run smaller-scale experiments to ablate factors that could influence attack success, including broader ratios of poisoned to clean data and non-random distributions of poisoned samples. Finally, we demonstrate the same dynamics for poisoning during fine-tuning. Altogether, our results suggest that injecting backdoors through data poisoning may be easier for large models than previously believed as the number of poisons required does not scale up with model size, highlighting the need for more research on defences to mitigate this risk in future models.

摘要: 中毒攻击可能会通过将恶意文档注入大型语言模型（LLM）的训练数据中来危及大型语言模型（LLM）的安全性。现有的工作已经研究了训练前中毒，假设对手控制了一定比例的训练素材。然而，对于大型模型来说，即使是很小的百分比也会转化为不切实际的大量数据。这项工作首次证明，无论数据集大小如何，中毒攻击都需要几乎恒定数量的文档。我们进行了迄今为止最大的预训练中毒实验，在龙猫最佳数据集（6 B至260 B代币）上预训练600 M至13 B参数的模型。我们发现，尽管最大的模型在干净数据上训练了20倍以上的数据，但250个有毒文档同样会损害所有模型和数据集大小的模型。我们还进行了较小规模的实验，以消除可能影响攻击成功的因素，包括更广泛的中毒数据与干净数据的比例以及中毒样本的非随机分布。最后，我们演示了微调期间中毒的相同动态。总而言之，我们的结果表明，对于大型模型来说，通过数据中毒注入后门可能比之前认为的更容易，因为所需的毒药数量不会随着模型大小而增加，这凸显了需要对防御进行更多研究，以减轻未来模型中的这种风险。



## **24. Sustainable Self-evolution Adversarial Training**

可持续自我进化对抗训练 cs.CV

Accepted to ACMMM 2024

**SubmitDate**: 2025-10-08    [abs](http://arxiv.org/abs/2412.02270v2) [paper-pdf](http://arxiv.org/pdf/2412.02270v2)

**Authors**: Wenxuan Wang, Chenglei Wang, Huihui Qi, Menghao Ye, Xuelin Qian, Peng Wang, Yanning Zhang

**Abstract**: With the wide application of deep neural network models in various computer vision tasks, there has been a proliferation of adversarial example generation strategies aimed at deeply exploring model security. However, existing adversarial training defense models, which rely on single or limited types of attacks under a one-time learning process, struggle to adapt to the dynamic and evolving nature of attack methods. Therefore, to achieve defense performance improvements for models in long-term applications, we propose a novel Sustainable Self-Evolution Adversarial Training (SSEAT) framework. Specifically, we introduce a continual adversarial defense pipeline to realize learning from various kinds of adversarial examples across multiple stages. Additionally, to address the issue of model catastrophic forgetting caused by continual learning from ongoing novel attacks, we propose an adversarial data replay module to better select more diverse and key relearning data. Furthermore, we design a consistency regularization strategy to encourage current defense models to learn more from previously trained ones, guiding them to retain more past knowledge and maintain accuracy on clean samples. Extensive experiments have been conducted to verify the efficacy of the proposed SSEAT defense method, which demonstrates superior defense performance and classification accuracy compared to competitors.Code is available at https://github.com/aup520/SSEAT

摘要: 随着深度神经网络模型在各种计算机视觉任务中的广泛应用，旨在深入探索模型安全性的对抗性示例生成策略不断涌现。然而，现有的对抗性训练防御模型依赖于一次性学习过程下的单一或有限类型的攻击，难以适应攻击方法的动态和不断发展的本质。因此，为了在长期应用中实现模型的防御性能改进，我们提出了一种新型的可持续自我进化对抗训练（SSEAT）框架。具体来说，我们引入了一个持续的对抗防御管道，以实现跨多个阶段从各种对抗示例中学习。此外，为了解决从正在进行的新颖攻击中持续学习所导致的模型灾难性遗忘问题，我们提出了一个对抗性数据重播模块，以更好地选择更多样化和关键的重新学习数据。此外，我们设计了一种一致性正规化策略，以鼓励当前的防御模型从之前训练的模型中学习更多信息，引导它们保留更多过去的知识并保持干净样本的准确性。已经进行了大量实验来验证所提出的SSEAT防御方法的有效性，该方法与竞争对手相比具有优越的防御性能和分类准确性。代码可在https://github.com/aup520/SSEAT上获取



## **25. Guardians of Image Quality: Benchmarking Defenses Against Adversarial Attacks on Image Quality Metrics**

图像质量守护者：针对图像质量预设的对抗攻击的基准防御 cs.CV

**SubmitDate**: 2025-10-08    [abs](http://arxiv.org/abs/2408.01541v2) [paper-pdf](http://arxiv.org/pdf/2408.01541v2)

**Authors**: Alexander Gushchin, Khaled Abud, Georgii Bychkov, Ekaterina Shumitskaya, Anna Chistyakova, Sergey Lavrushkin, Bader Rasheed, Kirill Malyshev, Dmitriy Vatolin, Anastasia Antsiferova

**Abstract**: In the field of Image Quality Assessment (IQA), the adversarial robustness of the metrics poses a critical concern. This paper presents a comprehensive benchmarking study of various defense mechanisms in response to the rise in adversarial attacks on IQA. We systematically evaluate 25 defense strategies, including adversarial purification, adversarial training, and certified robustness methods. We applied 14 adversarial attack algorithms of various types in both non-adaptive and adaptive settings and tested these defenses against them. We analyze the differences between defenses and their applicability to IQA tasks, considering that they should preserve IQA scores and image quality. The proposed benchmark aims to guide future developments and accepts submissions of new methods, with the latest results available online: https://videoprocessing.ai/benchmarks/iqa-defenses.html.

摘要: 在图像质量评估（IQA）领域，度量的对抗鲁棒性是一个关键问题。本文对各种防御机制进行了全面的基准测试研究，以应对对IQA的对抗性攻击的增加。我们系统地评估了25种防御策略，包括对抗性净化，对抗性训练和认证的鲁棒性方法。我们在非自适应和自适应设置中应用了14种不同类型的对抗性攻击算法，并测试了这些防御措施。我们分析了防御之间的差异和它们对IQA任务的适用性，认为它们应该保持IQA分数和图像质量。拟议的基准旨在指导未来的发展，并接受新方法的提交，最新结果可在线获得：https://videoprocessing.ai/benchmarks/iqa-defenses.html。



## **26. Universally Composable Termination Analysis of Tendermint**

Tendermint的通用组合终止分析 cs.CR

35 pages including references, 16 figures, 2 tables. Submitted to  ACNS 2026

**SubmitDate**: 2025-10-08    [abs](http://arxiv.org/abs/2510.01097v2) [paper-pdf](http://arxiv.org/pdf/2510.01097v2)

**Authors**: Zhixin Dong, Xian Xu, Yuhang Zeng, Mingchao Wan, Chunmiao Li

**Abstract**: Modern blockchain systems operating in adversarial environments require robust consensus protocols that guarantee both safety and termination under network delay attacks. Tendermint, a widely adopted consensus protocol in consortium blockchains, achieves high throughput and finality. However, previous analysis of the safety and termination has been done in a standalone fashion, with no consideration of the composition with other protocols interacting with it in a concurrent manner. Moreover, the termination properties under adaptive network delays caused by Byzantine adversaries have not been formally analyzed. This paper presents the first universally composable (UC) security analysis of Tendermint, demonstrating its resilience against strategic message-delay attacks. By constructing a UC ideal model of Tendermint, we formalize its core mechanisms: phase-base consensus procedure, dynamic timeouts, proposal locking, leader rotation, and others, under a network adversary that selectively delays protocol messages. Our main result proves that the Tendermint protocol UC-realizes the ideal Tendermint model, which ensures bounded termination latency, i.e., guaranteed termination, even when up to $f<n/3$ nodes are Byzantine (where $n$ is the number of nodes participating in the consensus), provided that network delays remain within a protocol-defined threshold under the partially synchronous net assumption. Specifically, through formal proofs within the UC framework, we show that Tendermint maintains safety and termination. By the composition theorem of UC, this guarantees that these properties are maintained when Tendermint is composed with various blockchain components.

摘要: 在对抗环境中运行的现代区块链系统需要强大的共识协议，以保证网络延迟攻击下的安全性和终止性。Tendermint是联盟区块链中广泛采用的共识协议，可实现高吞吐量和最终性。然而，之前对安全性和终止的分析是以独立的方式进行的，没有考虑与其他协议以并发的方式相互作用的组成。此外，拜占庭对手造成的自适应网络延迟下的终止特性尚未得到正式分析。本文介绍了Tendermint的第一个通用可组合（UC）安全分析，展示了其针对战略消息延迟攻击的弹性。通过构建Tendermint的UC理想模型，我们在选择性地延迟协议消息的网络对手下正式化了其核心机制：基于阶段的共识过程、动态超时、提案锁定、领导者轮换等。我们的主要结果证明，Tendermint协议UC实现了理想的Tendermint模型，该模型确保了有限的终止延迟，即保证终止，即使最多$f<n/3$个节点是拜占庭式的（其中$n$是参与共识的节点数量），前提是网络延迟保持在部分同步网络假设下协议定义的阈值内。具体来说，通过UC框架内的正式证明，我们表明Tendermint维护了安全性和终止性。根据UC的合成定理，这保证了当Tendermint由各种区块链组件组成时，这些属性得到维护。



## **27. DiffMI: Breaking Face Recognition Privacy via Diffusion-Driven Training-Free Model Inversion**

迪夫MI：通过扩散驱动免训练模型倒置打破面部识别隐私 cs.CR

**SubmitDate**: 2025-10-08    [abs](http://arxiv.org/abs/2504.18015v3) [paper-pdf](http://arxiv.org/pdf/2504.18015v3)

**Authors**: Hanrui Wang, Shuo Wang, Chun-Shien Lu, Isao Echizen

**Abstract**: Face recognition poses serious privacy risks due to its reliance on sensitive and immutable biometric data. While modern systems mitigate privacy risks by mapping facial images to embeddings (commonly regarded as privacy-preserving), model inversion attacks reveal that identity information can still be recovered, exposing critical vulnerabilities. However, existing attacks are often computationally expensive and lack generalization, especially those requiring target-specific training. Even training-free approaches suffer from limited identity controllability, hindering faithful reconstruction of nuanced or unseen identities. In this work, we propose DiffMI, the first diffusion-driven, training-free model inversion attack. DiffMI introduces a novel pipeline combining robust latent code initialization, a ranked adversarial refinement strategy, and a statistically grounded, confidence-aware optimization objective. DiffMI applies directly to unseen target identities and face recognition models, offering greater adaptability than training-dependent approaches while significantly reducing computational overhead. Our method achieves 84.42%--92.87% attack success rates against inversion-resilient systems and outperforms the best prior training-free GAN-based approach by 4.01%--9.82%. The implementation is available at https://github.com/azrealwang/DiffMI.

摘要: 由于面部识别依赖敏感且不可变的生物识别数据，因此存在严重的隐私风险。虽然现代系统通过将面部图像映射到嵌入（通常被认为是隐私保护）来降低隐私风险，但模型倒置攻击表明身份信息仍然可以恢复，从而暴露了关键漏洞。然而，现有的攻击通常计算昂贵且缺乏通用性，尤其是那些需要针对特定目标训练的攻击。即使是无需训练的方法也会受到有限的身份控制性的影响，阻碍了细致入微或不可见的身份的忠实重建。在这项工作中，我们提出了迪夫MI，这是第一个扩散驱动的、免训练的模型倒置攻击。迪夫MI引入了一种新型管道，将稳健的潜在代码初始化、排序对抗细化策略和基于统计的、信任度感知的优化目标结合在一起。迪夫MI直接适用于不可见的目标身份和人脸识别模型，比依赖训练的方法提供更大的适应性，同时显着减少计算负担。我们的方法针对反向弹性系统的攻击成功率达到了84.42%-92.87%，比之前最好的免训练基于GAN的方法高出4.01%-9.82%。该实现可在https://github.com/azrealwang/DiffMI上获取。



## **28. Minimal Cascade Gradient Smoothing for Fast Transferable Preemptive Adversarial Defense**

快速可转移先发制人对抗防御的最小级联梯度平滑 cs.CR

**SubmitDate**: 2025-10-08    [abs](http://arxiv.org/abs/2407.15524v8) [paper-pdf](http://arxiv.org/pdf/2407.15524v8)

**Authors**: Hanrui Wang, Ching-Chun Chang, Chun-Shien Lu, Ching-Chia Kao, Isao Echizen

**Abstract**: Adversarial attacks persist as a major challenge in deep learning. While training- and test-time defenses are well-studied, they often reduce clean accuracy, incur high cost, or fail under adaptive threats. In contrast, preemptive defenses, which perturb media before release, offer a practical alternative but remain slow, model-coupled, and brittle. We propose the Minimal Sufficient Preemptive Defense (MSPD), a fast, transferable framework that defends against future attacks without access to the target model or gradients. MSPD is driven by Minimal Cascade Gradient Smoothing (MCGS), a two-epoch optimization paradigm executed on a surrogate backbone. This defines a minimal yet effective regime for robust generalization across unseen models and attacks. MSPD runs at 0.02s/image (CIFAR-10) and 0.26s/image (ImageNet), 28--1696x faster than prior preemptive methods, while improving robust accuracy by +5% and clean accuracy by +3.7% across 11 models and 7 attacks. To evaluate adaptive robustness, we introduce Preemptive Reversion, the first white-box diagnostic attack that cancels preemptive perturbations under full gradient access. Even in this setting, MSPD retains a +2.2% robustness margin over the baseline. In practice, when gradients are unavailable, MSPD remains reliable and efficient. MSPD, MCGS, and Preemptive Reversion are each supported by formal theoretical proofs. The implementation is available at https://github.com/azrealwang/MSPD.

摘要: 对抗性攻击仍然是深度学习的一大挑战。虽然训练和测试时的防御已经得到了充分的研究，但它们通常会降低准确性、产生高成本或在适应性威胁下失败。相比之下，先发制人的防御在发布前扰乱媒体，提供了一种实用的替代方案，但仍然缓慢、模型耦合且脆弱。我们提出了最小充分先发制人防御（MSPD），这是一个快速、可转移的框架，可以在不访问目标模型或梯度的情况下抵御未来的攻击。MSPD由最小级联梯度平滑（MCGS）驱动，这是一种在代理主干上执行的两阶段优化范式。这定义了一个最小但有效的机制，用于对未见过的模型和攻击进行稳健概括。MSPD的运行速度为0.02s/图像（CIFAR-10）和0.26s/图像（ImageNet），比之前的抢先方法快28- 1696倍，同时在11个模型和7次攻击中将稳健准确性提高+5%，干净准确性提高+3.7%。为了评估自适应鲁棒性，我们引入了先发制人逆转，这是第一个白盒诊断攻击，可以在全梯度访问下取消先发制人扰动。即使在这种情况下，MSPD仍保持比基线+2.2%的稳健性利润。在实践中，当梯度不可用时，MSPD仍然可靠且高效。MSPD、MCGS和先发制人逆转均得到正式理论证明的支持。该实现可在https://github.com/azrealwang/MSPD上获取。



## **29. GreedyPixel: Fine-Grained Black-Box Adversarial Attack Via Greedy Algorithm**

GreedyPixel：通过贪婪算法进行细粒度黑匣子对抗攻击 cs.CV

**SubmitDate**: 2025-10-08    [abs](http://arxiv.org/abs/2501.14230v2) [paper-pdf](http://arxiv.org/pdf/2501.14230v2)

**Authors**: Hanrui Wang, Ching-Chun Chang, Chun-Shien Lu, Christopher Leckie, Isao Echizen

**Abstract**: Deep neural networks are highly vulnerable to adversarial examples that inputs with small, carefully crafted perturbations that cause misclassification, making adversarial attacks an essential tool for robustness evaluation. Existing black-box attacks fall into three categories: query-only, transfer-only, and query-and-transfer, and vary in perturbation pattern and optimization strategy. However, no prior method jointly achieves query-and-transfer guidance, pixel-wise sparsity, and training-free direct optimization, leaving a gap between black-box flexibility and white-box precision. We present GreedyPixel, a new attack framework that fills this gap by combining a surrogate-derived pixel priority map with greedy, per-pixel optimization refined by query feedback. This design reduces the exponential brute-force search space to a tractable linear procedure, guarantees monotonic loss decrease and convergence to a coordinate-wise optimum, and concentrates perturbations on robust, semantically meaningful pixels to improve perceptual quality. Extensive experiments on CIFAR-10 and ImageNet under both white-box and black-box settings demonstrate that GreedyPixel achieves state-of-the-art attack success rates and produces visually imperceptible perturbations. Our results show that GreedyPixel bridges the precision gap between white-box and black-box attacks and provides a practical framework for fine-grained robustness evaluation. The implementation is available at https://github.com/azrealwang/greedypixel.

摘要: 深度神经网络非常容易受到对抗性示例的影响，这些示例输入的是微小的、精心设计的扰动，从而导致错误分类，使对抗性攻击成为稳健性评估的重要工具。现有的黑匣子攻击分为三类：仅查询、仅传输和查询并传输，并且其扰动模式和优化策略各不相同。然而，没有任何现有方法能够共同实现查询和传输引导、像素稀疏性和免训练直接优化，从而在黑匣子灵活性和白盒精确性之间留下了差距。我们提出了GreedyPixel，这是一种新的攻击框架，它通过将代理衍生的像素优先级地图与由查询反馈改进的贪婪的每像素优化相结合来填补这一空白。该设计将指数暴力搜索空间简化为易于处理的线性过程，保证单调损失减少并收敛到坐标最优值，并将扰动集中在稳健、语义有意义的像素上，以提高感知质量。在白盒和黑匣子设置下对CIFAR-10和ImageNet进行的大量实验表明，GreedyPixel实现了最先进的攻击成功率，并产生视觉上难以感知的扰动。我们的结果表明，GreedyPixel弥合了白盒攻击和黑盒攻击之间的精度差距，并为细粒度稳健性评估提供了实用的框架。该实现可在www.example.com上获取。



## **30. RedTWIZ: Diverse LLM Red Teaming via Adaptive Attack Planning**

RedTWIZ：通过自适应攻击规划实现多元化LLM红色团队 cs.CR

**SubmitDate**: 2025-10-08    [abs](http://arxiv.org/abs/2510.06994v1) [paper-pdf](http://arxiv.org/pdf/2510.06994v1)

**Authors**: Artur Horal, Daniel Pina, Henrique Paz, Iago Paulo, João Soares, Rafael Ferreira, Diogo Tavares, Diogo Glória-Silva, João Magalhães, David Semedo

**Abstract**: This paper presents the vision, scientific contributions, and technical details of RedTWIZ: an adaptive and diverse multi-turn red teaming framework, to audit the robustness of Large Language Models (LLMs) in AI-assisted software development. Our work is driven by three major research streams: (1) robust and systematic assessment of LLM conversational jailbreaks; (2) a diverse generative multi-turn attack suite, supporting compositional, realistic and goal-oriented jailbreak conversational strategies; and (3) a hierarchical attack planner, which adaptively plans, serializes, and triggers attacks tailored to specific LLM's vulnerabilities. Together, these contributions form a unified framework -- combining assessment, attack generation, and strategic planning -- to comprehensively evaluate and expose weaknesses in LLMs' robustness. Extensive evaluation is conducted to systematically assess and analyze the performance of the overall system and each component. Experimental results demonstrate that our multi-turn adversarial attack strategies can successfully lead state-of-the-art LLMs to produce unsafe generations, highlighting the pressing need for more research into enhancing LLM's robustness.

摘要: 本文介绍了RedTWIZ的愿景、科学贡献和技术细节：一个自适应且多样化的多回合红色团队框架，用于审核大型语言模型（LLM）在人工智能辅助软件开发中的稳健性。我们的工作由三个主要研究流推动：（1）对LLM对话越狱的稳健和系统性评估;（2）多元化的生成式多回合攻击套件，支持组合性、现实性和面向目标的越狱对话策略;（3）分层攻击规划器，它自适应地规划、序列化和触发针对特定LLM漏洞的攻击。这些贡献共同构成了一个统一的框架--结合了评估、攻击生成和战略规划--以全面评估和揭露LLM稳健性的弱点。进行广泛的评估，以系统地评估和分析整个系统和每个组件的性能。实验结果表明，我们的多回合对抗攻击策略可以成功导致最先进的LLM产生不安全的世代，凸显了对增强LLM稳健性进行更多研究的迫切需要。



## **31. OBJVanish: Physically Realizable Text-to-3D Adv. Generation of LiDAR-Invisible Objects**

ObJVanish：LiDART隐形物体的物理可实现文本到3D高级生成 cs.CV

**SubmitDate**: 2025-10-08    [abs](http://arxiv.org/abs/2510.06952v1) [paper-pdf](http://arxiv.org/pdf/2510.06952v1)

**Authors**: Bing Li, Wuqi Wang, Yanan Zhang, Jingzheng Li, Haigen Min, Wei Feng, Xingyu Zhao, Jie Zhang, Qing Guo

**Abstract**: LiDAR-based 3D object detectors are fundamental to autonomous driving, where failing to detect objects poses severe safety risks. Developing effective 3D adversarial attacks is essential for thoroughly testing these detection systems and exposing their vulnerabilities before real-world deployment. However, existing adversarial attacks that add optimized perturbations to 3D points have two critical limitations: they rarely cause complete object disappearance and prove difficult to implement in physical environments. We introduce the text-to-3D adversarial generation method, a novel approach enabling physically realizable attacks that can generate 3D models of objects truly invisible to LiDAR detectors and be easily realized in the real world. Specifically, we present the first empirical study that systematically investigates the factors influencing detection vulnerability by manipulating the topology, connectivity, and intensity of individual pedestrian 3D models and combining pedestrians with multiple objects within the CARLA simulation environment. Building on the insights, we propose the physically-informed text-to-3D adversarial generation (Phy3DAdvGen) that systematically optimizes text prompts by iteratively refining verbs, objects, and poses to produce LiDAR-invisible pedestrians. To ensure physical realizability, we construct a comprehensive object pool containing 13 3D models of real objects and constrain Phy3DAdvGen to generate 3D objects based on combinations of objects in this set. Extensive experiments demonstrate that our approach can generate 3D pedestrians that evade six state-of-the-art (SOTA) LiDAR 3D detectors in both CARLA simulation and physical environments, thereby highlighting vulnerabilities in safety-critical applications.

摘要: 基于LiDART的3D物体检测器是自动驾驶的基础，无法检测到物体会带来严重的安全风险。开发有效的3D对抗攻击对于彻底测试这些检测系统并在现实世界部署之前暴露其漏洞至关重要。然而，现有的为3D点添加优化扰动的对抗攻击有两个关键局限性：它们很少导致对象完全消失，并且很难在物理环境中实现。我们引入了文本到3D对抗生成方法，这是一种新颖的方法，能够实现物理上可实现的攻击，可以生成LiDART检测器真正不可见的对象的3D模型，并且可以轻松在现实世界中实现。具体来说，我们提出了第一项实证研究，该研究通过操纵单个行人3D模型的布局、连接性和强度并将行人与CARLA模拟环境中的多个对象相结合来系统地调查影响检测脆弱性的因素。在这些见解的基础上，我们提出了基于物理信息的文本到3D对抗生成（Phy 3DAdvGen），该生成通过迭代细化动词、对象和姿势来系统地优化文本提示，以生成LiDART不可见的行人。为了确保物理可实现性，我们构建了一个包含13个真实对象3D模型的综合对象池，并约束Phy 3DAdvGen基于此集中对象的组合生成3D对象。大量实验表明，我们的方法可以在CARLA模拟和物理环境中生成避开六个最先进（SOTA）LiDART 3D检测器的3D行人，从而凸显了安全关键应用中的漏洞。



## **32. Get RICH or Die Scaling: Profitably Trading Inference Compute for Robustness**

获取财富或死亡缩放：盈利交易推理计算以实现稳健性 cs.LG

17 pages

**SubmitDate**: 2025-10-08    [abs](http://arxiv.org/abs/2510.06790v1) [paper-pdf](http://arxiv.org/pdf/2510.06790v1)

**Authors**: Tavish McDonald, Bo Lei, Stanislav Fort, Bhavya Kailkhura, Brian Bartoldson

**Abstract**: Models are susceptible to adversarially out-of-distribution (OOD) data despite large training-compute investments into their robustification. Zaremba et al. (2025) make progress on this problem at test time, showing LLM reasoning improves satisfaction of model specifications designed to thwart attacks, resulting in a correlation between reasoning effort and robustness to jailbreaks. However, this benefit of test compute fades when attackers are given access to gradients or multimodal inputs. We address this gap, clarifying that inference-compute offers benefits even in such cases. Our approach argues that compositional generalization, through which OOD data is understandable via its in-distribution (ID) components, enables adherence to defensive specifications on adversarially OOD inputs. Namely, we posit the Robustness from Inference Compute Hypothesis (RICH): inference-compute defenses profit as the model's training data better reflects the attacked data's components. We empirically support this hypothesis across vision language model and attack types, finding robustness gains from test-time compute if specification following on OOD data is unlocked by compositional generalization, while RL finetuning and protracted reasoning are not critical. For example, increasing emphasis on defensive specifications via prompting lowers the success rate of gradient-based multimodal attacks on VLMs robustified by adversarial pretraining, but this same intervention provides no such benefit to not-robustified models. This correlation of inference-compute's robustness benefit with base model robustness is the rich-get-richer dynamic of the RICH: attacked data components are more ID for robustified models, aiding compositional generalization to OOD data. Accordingly, we advise layering train-time and test-time defenses to obtain their synergistic benefit.

摘要: 尽管模型的鲁棒性投入了大量的训练计算投资，但它们仍然容易受到不利的分布外（OOD）数据的影响。Zaremba等人（2025）在测试时在这个问题上取得了进展，表明LLM推理提高了旨在阻止攻击的模型规范的满意度，从而导致推理工作量和越狱稳健性之间的相关性。然而，当攻击者能够访问梯度或多模式输入时，测试计算的这种好处就会消失。我们解决了这一差距，澄清了即使在这种情况下，推理计算也能带来好处。我们的方法认为，组合概括（OOD数据可以通过其内分布（ID）组件来理解）使得能够遵守针对敌对OOD输入的防御规范。也就是说，我们从推理计算假设（RICH）中验证了鲁棒性：由于模型的训练数据更好地反映了受攻击数据的成分，推理计算防御会获利。我们在视觉语言模型和攻击类型中从经验上支持了这一假设，如果OOD数据上的规范通过组合概括解锁，则可以从测试时计算中找到鲁棒性收益，而RL微调和持久推理并不关键。例如，通过提示来增加对防御规范的强调会降低对由对抗性预训练稳健的VLM的基于梯度的多模式攻击的成功率，但同样的干预并没有为非稳健的模型提供这样的好处。推理计算的鲁棒性与基础模型鲁棒性的这种相关性是RICH的丰富-越来越丰富的动态：受攻击的数据组件对于鲁棒模型来说更具ID，有助于组合泛化到OOD数据。因此，我们建议分层训练时和测试时的防御，以获得其协同效益。



## **33. Benchmarking Gaslighting Negation Attacks Against Multimodal Large Language Models**

针对多模式大型语言模型的Gaslighting否定攻击基准 cs.CL

Project website:  https://yxg1005.github.io/GaslightingNegationAttacks/

**SubmitDate**: 2025-10-08    [abs](http://arxiv.org/abs/2501.19017v4) [paper-pdf](http://arxiv.org/pdf/2501.19017v4)

**Authors**: Bin Zhu, Yinxuan Gui, Huiyan Qi, Jingjing Chen, Chong-Wah Ngo, Ee-Peng Lim

**Abstract**: Multimodal Large Language Models (MLLMs) have exhibited remarkable advancements in integrating different modalities, excelling in complex understanding and generation tasks. Despite their success, MLLMs remain vulnerable to conversational adversarial inputs. In this paper, we systematically study gaslighting negation attacks: a phenomenon where models, despite initially providing correct answers, are persuaded by user-provided negations to reverse their outputs, often fabricating justifications. We conduct extensive evaluations of state-of-the-art MLLMs across diverse benchmarks and observe substantial performance drops when negation is introduced. Notably, we introduce the first benchmark GaslightingBench, specifically designed to evaluate the vulnerability of MLLMs to negation arguments. GaslightingBench consists of multiple-choice questions curated from existing datasets, along with generated negation prompts across 20 diverse categories. Throughout extensive evaluation, we find that proprietary models such as Gemini-1.5-flash and GPT-4o demonstrate better resilience compared to open-source counterparts like Qwen2-VL and LLaVA, though even advanced reasoning-oriented models like Gemini-2.5-Pro remain susceptible. Our category-level analysis further shows that subjective or socially nuanced domains (e.g., Social Relation, Image Emotion) are especially fragile, while more objective domains (e.g., Geography) exhibit relatively smaller but still notable drops. Overall, all evaluated MLLMs struggle to maintain logical consistency under gaslighting negation attack. These findings highlight a fundamental robustness gap and provide insights for developing more reliable and trustworthy multimodal AI systems. Project website: https://yxg1005.github.io/GaslightingNegationAttacks/.

摘要: 多模式大型语言模型（MLLM）在集成不同模式方面表现出了显着的进步，在复杂的理解和生成任务中表现出色。尽管取得了成功，MLLM仍然容易受到对话对抗输入的影响。在本文中，我们系统地研究了煤气灯否定攻击：这是一种现象，模型尽管最初提供了正确的答案，但被用户提供的否定说服来扭转其输出，通常编造理由。我们对不同基准的最先进的MLLM进行了广泛评估，并观察到当引入否定时性能会大幅下降。值得注意的是，我们引入了第一个基准GaslightingBench，专门用于评估MLLM对否定论点的脆弱性。GaslightingBench由根据现有数据集精心设计的多项选择题以及生成的跨越20个不同类别的否定提示组成。在广泛的评估中，我们发现Gemini-1.5-Flash和GPT-4 o等专有模型与Qwen 2-BL和LLaVA等开源模型相比表现出更好的弹性，尽管即使是像Gemini-2.5-Pro这样的高级推理导向模型仍然容易受到影响。我们的类别级分析进一步表明，主观或社会细微差别领域（例如，社会关系，形象情感）尤其脆弱，而更客观的领域（例如，地理）显示相对较小，但仍然显着下降。总体而言，所有评估的MLLM努力保持逻辑一致性下gaslighting否定攻击。这些发现突出了一个基本的鲁棒性差距，并为开发更可靠和值得信赖的多模态AI系统提供了见解。项目网站：https://yxg1005.github.io/GaslightingNegationAttacks/。



## **34. Towards the Worst-case Robustness of Large Language Models**

走向大型语言模型的最坏情况稳健性 cs.LG

**SubmitDate**: 2025-10-08    [abs](http://arxiv.org/abs/2501.19040v4) [paper-pdf](http://arxiv.org/pdf/2501.19040v4)

**Authors**: Huanran Chen, Yinpeng Dong, Zeming Wei, Hang Su, Jun Zhu

**Abstract**: Recent studies have revealed the vulnerability of large language models to adversarial attacks, where adversaries craft specific input sequences to induce harmful, violent, private, or incorrect outputs. In this work, we study their worst-case robustness, i.e., whether an adversarial example exists that leads to such undesirable outputs. We upper bound the worst-case robustness using stronger white-box attacks, indicating that most current deterministic defenses achieve nearly 0\% worst-case robustness. We propose a general tight lower bound for randomized smoothing using fractional knapsack solvers or 0-1 knapsack solvers, and using them to bound the worst-case robustness of all stochastic defenses. Based on these solvers, we provide theoretical lower bounds for several previous empirical defenses. For example, we certify the robustness of a specific case, smoothing using a uniform kernel, against \textit{any possible attack} with an average $\ell_0$ perturbation of 2.02 or an average suffix length of 6.41.

摘要: 最近的研究揭示了大型语言模型容易受到对抗攻击，对手会精心设计特定的输入序列来引发有害、暴力、私密或错误的输出。在这项工作中，我们研究了它们的最坏情况稳健性，即是否存在导致此类不良结果的对抗性例子。我们使用更强的白盒攻击来对最坏情况的稳健性进行上限，这表明当前大多数确定性防御实现了近0%的最坏情况的稳健性。我们提出了使用分数背包求解器或0-1背包求解器的随机平滑的一般紧下界，并使用它们来限制所有随机防御的最坏情况稳健性。基于这些求解器，我们为之前的几个经验防御提供了理论下限。例如，我们证明了特定情况的稳健性，使用统一核进行平滑，针对\texttit {任何可能的攻击}，平均$\ell_0 $扰动为2.02或平均后缀长度为6.41。



## **35. SafeGuider: Robust and Practical Content Safety Control for Text-to-Image Models**

SafeGuider：针对文本到图像模型的稳健且实用的内容安全控制 cs.CR

Accepted by ACM CCS 2025

**SubmitDate**: 2025-10-08    [abs](http://arxiv.org/abs/2510.05173v2) [paper-pdf](http://arxiv.org/pdf/2510.05173v2)

**Authors**: Peigui Qi, Kunsheng Tang, Wenbo Zhou, Weiming Zhang, Nenghai Yu, Tianwei Zhang, Qing Guo, Jie Zhang

**Abstract**: Text-to-image models have shown remarkable capabilities in generating high-quality images from natural language descriptions. However, these models are highly vulnerable to adversarial prompts, which can bypass safety measures and produce harmful content. Despite various defensive strategies, achieving robustness against attacks while maintaining practical utility in real-world applications remains a significant challenge. To address this issue, we first conduct an empirical study of the text encoder in the Stable Diffusion (SD) model, which is a widely used and representative text-to-image model. Our findings reveal that the [EOS] token acts as a semantic aggregator, exhibiting distinct distributional patterns between benign and adversarial prompts in its embedding space. Building on this insight, we introduce \textbf{SafeGuider}, a two-step framework designed for robust safety control without compromising generation quality. SafeGuider combines an embedding-level recognition model with a safety-aware feature erasure beam search algorithm. This integration enables the framework to maintain high-quality image generation for benign prompts while ensuring robust defense against both in-domain and out-of-domain attacks. SafeGuider demonstrates exceptional effectiveness in minimizing attack success rates, achieving a maximum rate of only 5.48\% across various attack scenarios. Moreover, instead of refusing to generate or producing black images for unsafe prompts, \textbf{SafeGuider} generates safe and meaningful images, enhancing its practical utility. In addition, SafeGuider is not limited to the SD model and can be effectively applied to other text-to-image models, such as the Flux model, demonstrating its versatility and adaptability across different architectures. We hope that SafeGuider can shed some light on the practical deployment of secure text-to-image systems.

摘要: 文本到图像模型在从自然语言描述生成高质量图像方面表现出了非凡的能力。然而，这些模型非常容易受到对抗提示的影响，这可能会绕过安全措施并产生有害内容。尽管有各种防御策略，但在现实世界应用程序中保持实用性的同时实现针对攻击的鲁棒性仍然是一个重大挑战。为了解决这个问题，我们首先对稳定扩散（SD）模型中的文本编码器进行了实证研究，该模型是一种广泛使用且具有代表性的文本到图像模型。我们的研究结果表明，[EOS]令牌充当语义聚合器，在其嵌入空间中的良性提示和对抗提示之间表现出明显的分布模式。基于这一见解，我们引入了\textBF{SafeGuider}，这是一个两步框架，旨在在不影响发电质量的情况下进行稳健的安全控制。SafeGuider将嵌入级识别模型与安全意识特征擦除束搜索算法相结合。此集成使该框架能够为良性提示维持高质量图像生成，同时确保针对域内和域外攻击的强大防御。SafeGuider在最大限度地降低攻击成功率方面表现出出色的有效性，在各种攻击场景中实现的最高攻击成功率仅为5.48%。此外，\textBF{SafeGuider}不会拒绝为不安全提示生成或产生黑色图像，而是生成安全且有意义的图像，增强了其实际实用性。此外，SafeGuider不限于SD模型，可以有效应用于其他文本到图像模型，例如Flux模型，展示了其在不同架构中的通用性和适应性。我们希望SafeGuider能够为安全文本到图像系统的实际部署提供一些线索。



## **36. Do Internal Layers of LLMs Reveal Patterns for Jailbreak Detection?**

LLM的内层是否揭示了越狱检测的模式？ cs.CL

**SubmitDate**: 2025-10-08    [abs](http://arxiv.org/abs/2510.06594v1) [paper-pdf](http://arxiv.org/pdf/2510.06594v1)

**Authors**: Sri Durga Sai Sowmya Kadali, Evangelos E. Papalexakis

**Abstract**: Jailbreaking large language models (LLMs) has emerged as a pressing concern with the increasing prevalence and accessibility of conversational LLMs. Adversarial users often exploit these models through carefully engineered prompts to elicit restricted or sensitive outputs, a strategy widely referred to as jailbreaking. While numerous defense mechanisms have been proposed, attackers continuously develop novel prompting techniques, and no existing model can be considered fully resistant. In this study, we investigate the jailbreak phenomenon by examining the internal representations of LLMs, with a focus on how hidden layers respond to jailbreak versus benign prompts. Specifically, we analyze the open-source LLM GPT-J and the state-space model Mamba2, presenting preliminary findings that highlight distinct layer-wise behaviors. Our results suggest promising directions for further research on leveraging internal model dynamics for robust jailbreak detection and defense.

摘要: 随着对话式LLM的日益普及和可访问性，越狱大型语言模型（LLM）已成为一个紧迫的问题。敌对用户经常通过精心设计的提示来利用这些模型来获取受限或敏感的输出，这种策略被广泛称为越狱。虽然已经提出了许多防御机制，但攻击者不断开发新颖的提示技术，并且没有任何现有模型可以被认为是完全抵抗的。在这项研究中，我们通过检查LLM的内部表示来调查越狱现象，重点关注隐藏层如何对越狱与良性提示做出反应。具体来说，我们分析了开源LLM GPT-J和状态空间模型Mamba 2，提供了突出不同分层行为的初步发现。我们的结果为进一步研究利用内部模型动态来进行稳健的越狱检测和防御指明了有希望的方向。



## **37. MM-PoisonRAG: Disrupting Multimodal RAG with Local and Global Poisoning Attacks**

MM-PoisonRAG：通过本地和全球中毒攻击扰乱多模式RAG cs.LG

Code is available at https://github.com/HyeonjeongHa/MM-PoisonRAG

**SubmitDate**: 2025-10-08    [abs](http://arxiv.org/abs/2502.17832v3) [paper-pdf](http://arxiv.org/pdf/2502.17832v3)

**Authors**: Hyeonjeong Ha, Qiusi Zhan, Jeonghwan Kim, Dimitrios Bralios, Saikrishna Sanniboina, Nanyun Peng, Kai-Wei Chang, Daniel Kang, Heng Ji

**Abstract**: Multimodal large language models with Retrieval Augmented Generation (RAG) have significantly advanced tasks such as multimodal question answering by grounding responses in external text and images. This grounding improves factuality, reduces hallucination, and extends reasoning beyond parametric knowledge. However, this reliance on external knowledge poses a critical yet underexplored safety risk: knowledge poisoning attacks, where adversaries deliberately inject adversarial multimodal content into external knowledge bases to steer model toward generating incorrect or even harmful responses. To expose such vulnerabilities, we propose MM-PoisonRAG, the first framework to systematically design knowledge poisoning in multimodal RAG. We introduce two complementary attack strategies: Localized Poisoning Attack (LPA), which implants targeted multimodal misinformation to manipulate specific queries, and Globalized Poisoning Attack (GPA), which inserts a single adversarial knowledge to broadly disrupt reasoning and induce nonsensical responses across all queries. Comprehensive experiments across tasks, models, and access settings show that LPA achieves targeted manipulation with attack success rates of up to 56%, while GPA completely disrupts model generation to 0% accuracy with just a single adversarial knowledge injection. Our results reveal the fragility of multimodal RAG and highlight the urgent need for defenses against knowledge poisoning.

摘要: 具有检索增强生成（RAG）的多模态大型语言模型具有显着的高级任务，例如通过外部文本和图像中的基础响应进行多模态问题回答。这种基础提高了真实性，减少了幻觉，并将推理扩展到参数知识之外。然而，这种对外部知识的依赖带来了一个关键但尚未得到充分研究的安全风险：知识中毒攻击，其中对手故意将对抗性多模态内容注入外部知识库，以引导模型生成错误甚至有害的响应。为了暴露这些漏洞，我们提出了MM-PoisonRAG，第一个框架，系统地设计知识中毒的多模式RAG。我们介绍了两种互补的攻击策略：局部中毒攻击（LPA），它植入有针对性的多模态错误信息来操纵特定的查询，和全局中毒攻击（GPA），它插入一个单一的对抗性知识来广泛地破坏推理，并在所有查询中诱导无意义的响应。跨任务、模型和访问设置的综合实验表明，LPA实现了有针对性的操作，攻击成功率高达56%，而GPA仅用一次对抗性知识注入就完全破坏了模型生成，准确率为0%。我们的研究结果揭示了多模态RAG的脆弱性，并强调了迫切需要防御知识中毒。



## **38. Text-to-Image Models Leave Identifiable Signatures: Implications for Leaderboard Security**

文本到图像模型留下可识别签名：对排行榜安全性的影响 cs.LG

Accepted at Lock-LLM Workshop, NeurIPS 2025

**SubmitDate**: 2025-10-07    [abs](http://arxiv.org/abs/2510.06525v1) [paper-pdf](http://arxiv.org/pdf/2510.06525v1)

**Authors**: Ali Naseh, Anshuman Suri, Yuefeng Peng, Harsh Chaudhari, Alina Oprea, Amir Houmansadr

**Abstract**: Generative AI leaderboards are central to evaluating model capabilities, but remain vulnerable to manipulation. Among key adversarial objectives is rank manipulation, where an attacker must first deanonymize the models behind displayed outputs -- a threat previously demonstrated and explored for large language models (LLMs). We show that this problem can be even more severe for text-to-image leaderboards, where deanonymization is markedly easier. Using over 150,000 generated images from 280 prompts and 19 diverse models spanning multiple organizations, architectures, and sizes, we demonstrate that simple real-time classification in CLIP embedding space identifies the generating model with high accuracy, even without prompt control or historical data. We further introduce a prompt-level separability metric and identify prompts that enable near-perfect deanonymization. Our results indicate that rank manipulation in text-to-image leaderboards is easier than previously recognized, underscoring the need for stronger defenses.

摘要: 生成性人工智能排行榜是评估模型能力的核心，但仍然容易受到操纵。关键的对抗目标之一是排名操纵，攻击者必须首先对显示输出背后的模型进行去匿名化--这是之前针对大型语言模型（LLM）演示和探索的威胁。我们表明，对于文本到图像排行榜来说，这个问题可能会更加严重，因为其中的去匿名化明显更容易。使用从280个提示生成的超过150，000个图像和跨越多种组织、架构和规模的19个不同模型，我们证明了CLIP嵌入空间中的简单实时分类可以高准确性地识别生成模型，即使没有提示控制或历史数据。我们进一步引入预算级别的可分离性指标并识别能够实现近乎完美的去匿名化的提示。我们的结果表明，文本到图像排行榜中的排名操纵比之前认识到的更容易，这凸显了对更强防御的必要性。



## **39. Attacking the Spike: On the Transferability and Security of Spiking Neural Networks to Adversarial Examples**

攻击尖峰：关于尖峰神经网络到对抗性示例的可移植性和安全性 cs.NE

Accepted manuscript. Published in *Neurocomputing*, Volume 656, 2025,  Article 131506. Available online 12 September 2025. DOI:  10.1016/j.neucom.2025.131506

**SubmitDate**: 2025-10-07    [abs](http://arxiv.org/abs/2209.03358v4) [paper-pdf](http://arxiv.org/pdf/2209.03358v4)

**Authors**: Nuo Xu, Kaleel Mahmood, Haowen Fang, Ethan Rathbun, Caiwen Ding, Wujie Wen

**Abstract**: Spiking neural networks (SNNs) have drawn much attention for their high energy efficiency and recent advances in classification performance. However, unlike traditional deep learning, the robustness of SNNs to adversarial examples remains underexplored. This work advances the adversarial attack side of SNNs and makes three major contributions. First, we show that successful white-box attacks on SNNs strongly depend on the surrogate gradient estimation technique, even for adversarially trained models. Second, using the best single surrogate gradient estimator, we study the transferability of adversarial examples between SNNs and state-of-the-art architectures such as Vision Transformers (ViTs) and CNNs. Our analysis reveals two major gaps: no existing white-box attack leverages multiple surrogate estimators, and no single attack effectively fools both SNNs and non-SNN models simultaneously. Third, we propose the Mixed Dynamic Spiking Estimation (MDSE) attack, which dynamically combines multiple surrogate gradients to overcome these gaps. MDSE produces adversarial examples that fool both SNN and non-SNN models, achieving up to 91.4% higher effectiveness on SNN/ViT ensembles and a 3x boost on adversarially trained SNN ensembles over Auto-PGD. Experiments span three datasets (CIFAR-10, CIFAR-100, ImageNet) and nineteen classifiers, and we will release code and models upon publication.

摘要: 尖峰神经网络（SNN）因其高能效和分类性能的最新进展而引起了广泛关注。然而，与传统深度学习不同的是，SNN对对抗性示例的稳健性仍然没有得到充分的研究。这项工作推进了SNN的对抗攻击方面，并做出了三个主要贡献。首先，我们表明，对SNN的成功白盒攻击强烈依赖于代理梯度估计技术，即使对于敌对训练的模型也是如此。其次，使用最好的单一替代梯度估计器，我们研究SNN与Vision Transformers（ViTS）和CNN等最先进架构之间对抗性示例的可移植性。我们的分析揭示了两个主要差距：没有现有的白盒攻击利用多个代理估计器，也没有单一攻击有效地同时愚弄SNN和非SNN模型。第三，我们提出了混合动态尖峰估计（MDSE）攻击，它动态地组合多个替代梯度来克服这些差距。MDSE生成的对抗性示例可以愚弄SNN和非SNN模型，使SNN/ViT集成的有效性提高了91.4%，并使对抗训练的SNN集成的有效性提高了Auto-PVD的3倍。实验跨越三个数据集（CIFAR-10、CIFAR-100、ImageNet）和十九个分类器，我们将在发布后发布代码和模型。



## **40. Adversarial Surrogate Risk Bounds for Binary Classification**

二元分类的对抗性代理风险界限 cs.LG

37 pages, 3 figures

**SubmitDate**: 2025-10-07    [abs](http://arxiv.org/abs/2506.09348v2) [paper-pdf](http://arxiv.org/pdf/2506.09348v2)

**Authors**: Natalie S. Frank

**Abstract**: A central concern in classification is the vulnerability of machine learning models to adversarial attacks. Adversarial training is one of the most popular techniques for training robust classifiers, which involves minimizing an adversarial surrogate risk. Recent work has characterized the conditions under which any sequence minimizing the adversarial surrogate risk also minimizes the adversarial classification risk in the binary setting, a property known as adversarial consistency. However, these results do not address the rate at which the adversarial classification risk approaches its optimal value along such a sequence. This paper provides surrogate risk bounds that quantify that convergence rate.

摘要: 分类中的一个核心问题是机器学习模型对对抗攻击的脆弱性。对抗性训练是训练稳健分类器最流行的技术之一，它涉及最大限度地减少对抗性代理风险。最近的工作描述了任何使对抗性替代风险最小化的序列也使二元环境中的对抗性分类风险最小化的条件，这一属性称为对抗性一致性。然而，这些结果并没有解决对抗分类风险沿着这样的序列接近其最佳值的速度。本文提供了量化收敛率的替代风险界限。



## **41. LLM Unlearning via Neural Activation Redirection**

LLM通过神经激活重定向消除学习 cs.LG

**SubmitDate**: 2025-10-07    [abs](http://arxiv.org/abs/2502.07218v2) [paper-pdf](http://arxiv.org/pdf/2502.07218v2)

**Authors**: William F. Shen, Xinchi Qiu, Meghdad Kurmanji, Alex Iacob, Lorenzo Sani, Yihong Chen, Nicola Cancedda, Nicholas D. Lane

**Abstract**: The ability to selectively remove knowledge from LLMs is highly desirable. However, existing methods often struggle with balancing unlearning efficacy and retain model utility, and lack controllability at inference time to emulate base model behavior as if it had never seen the unlearned data. In this paper, we propose LUNAR, a novel unlearning method grounded in the Linear Representation Hypothesis and operates by redirecting the representations of unlearned data to activation regions that expresses its inability to answer. We show that contrastive features are not a prerequisite for effective activation redirection, and LUNAR achieves state-of-the-art unlearning performance and superior controllability. Specifically, LUNAR achieves between 2.9x and 11.7x improvement in the combined unlearning efficacy and model utility score (Deviation Score) across various base models and generates coherent, contextually appropriate responses post-unlearning. Moreover, LUNAR effectively reduces parameter updates to a single down-projection matrix, a novel design that significantly enhances efficiency by 20x and robustness. Finally, we demonstrate that LUNAR is robust to white-box adversarial attacks and versatile in real-world scenarios, including handling sequential unlearning requests.

摘要: 非常希望能够从LLM中选择性地删除知识。然而，现有的方法经常难以平衡未学习功效和保留模型效用，并且在推理时缺乏可控性，无法模拟基本模型行为，就好像它从未见过未学习的数据一样。在本文中，我们提出了LUNAR，这是一种基于线性表示假设的新型去学习方法，通过将未学习数据的表示重定向到表达其无法回答的激活区域来运作。我们表明，对比性特征并不是有效激活重定向的先决条件，而LUNAR实现了最先进的去学习性能和卓越的可控性。具体而言，LUNAR在各种基本模型中的综合取消学习功效和模型效用分数（偏差分数）方面实现了2.9倍至11.7倍的提高，并在取消学习后生成连贯、适合上下文的响应。此外，LUNAR有效地将参数更新减少到单个下投影矩阵，这是一种新颖的设计，可将效率和稳健性显着提高20倍。最后，我们证明了LUNAR对白盒对抗攻击具有鲁棒性，并且在现实世界场景中具有通用性，包括处理顺序取消学习请求。



## **42. Breaking Precision Time: OS Vulnerability Exploits Against IEEE 1588**

突破精确时间：针对IEEE 1588的操作系统漏洞利用 cs.CR

Published in IEEE ISPCS 2025

**SubmitDate**: 2025-10-07    [abs](http://arxiv.org/abs/2510.06421v1) [paper-pdf](http://arxiv.org/pdf/2510.06421v1)

**Authors**: Muhammad Abdullah Soomro, Fatima Muhammad Anwar

**Abstract**: The Precision Time Protocol (PTP), standardized as IEEE 1588, provides sub-microsecond synchronization across distributed systems and underpins critical infrastructure in telecommunications, finance, power systems, and industrial automation. While prior work has extensively analyzed PTP's vulnerability to network-based attacks, prompting the development of cryptographic protections and anomaly detectors, these defenses presume an uncompromised host. In this paper, we identify and exploit a critical blind spot in current threat models: kernel-level adversaries operating from within the host running the PTP stack. We present the first systematic study of kernel-rooted attacks on PTP, demonstrating how privileged attackers can manipulate system time by corrupting key interfaces without altering PTP network traffic. We implement three attack primitives, constant offset, progressive skew, and random jitter, using in-kernel payloads, and evaluate their impact on the widely used ptp4l and phc2sys daemons. Our experiments reveal that these attacks can silently destabilize clock synchronization, bypassing existing PTP security extensions. These findings highlight the urgent need to reconsider host-level trust assumptions and integrate kernel integrity into the design of secure time synchronization systems.

摘要: 精确时间协议（TTP）标准化为IEEE 1588，提供跨分布式系统的亚微秒同步，并支撑电信、金融、电力系统和工业自动化领域的关键基础设施。虽然之前的工作广泛分析了TTP对基于网络的攻击的脆弱性，促进了加密保护和异常检测器的开发，但这些防御假设是一个未受攻击的主机。在本文中，我们识别并利用当前威胁模型中的一个关键盲点：核心级对手在运行TTP堆栈的主机内操作。我们首次对TTP的基于核心的攻击进行了系统性研究，展示了特权攻击者如何通过破坏关键接口而不改变TTP网络流量来操纵系统时间。我们实现了三个攻击原语，恒定偏移，渐进偏斜和随机抖动，使用内核有效载荷，并评估其对广泛使用的ptp4l和phc2sys守护进程的影响。我们的实验表明，这些攻击可以悄悄地破坏时钟同步，绕过现有的PTP安全扩展。这些发现突出了迫切需要重新考虑主机级的信任假设和集成内核完整性的安全时间同步系统的设计。



## **43. When Should Selfish Miners Double-Spend?**

自私的矿工何时应该加倍花钱？ cs.CR

**SubmitDate**: 2025-10-07    [abs](http://arxiv.org/abs/2501.03227v2) [paper-pdf](http://arxiv.org/pdf/2501.03227v2)

**Authors**: Mustafa Doger, Sennur Ulukus

**Abstract**: Conventional double-spending attack models ignore the revenue losses stemming from the orphan blocks. On the other hand, selfish mining literature usually ignores the chance of the attacker to double-spend at no-cost in each attack cycle. In this paper, we give a rigorous stochastic analysis of an attack where the goal of the adversary is to double-spend while mining selfishly. To do so, we first combine stubborn and selfish mining attacks, i.e., construct a strategy where the attacker acts stubborn until its private branch reaches a certain length and then switches to act selfish. We provide the optimal stubbornness for each parameter regime. Next, we provide the maximum stubbornness that is still more profitable than honest mining and argue a connection between the level of stubbornness and the $k$-confirmation rule. We show that, at each attack cycle, if the level of stubbornness is higher than $k$, the adversary gets a free shot at double-spending. At each cycle, for a given stubbornness level, we rigorously formulate how great the probability of double-spending is. We further modify the attack in the stubborn regime in order to conceal the attack and increase the double-spending probability.

摘要: 传统的双重支出攻击模型忽略了孤儿区块带来的收入损失。另一方面，自私的采矿文献通常忽视攻击者在每个攻击周期中免费重复支出的机会。本文中，我们对攻击进行了严格的随机分析，其中对手的目标是在自私地挖掘的同时进行双重支出。为此，我们首先结合顽固和自私的采矿攻击，即构建一个策略，让攻击者表现得顽固，直到其私人分支达到一定长度，然后转向自私。我们为每个参数制度提供最佳的确定性。接下来，我们提供了仍然比诚实采矿更有利可图的最大顽固度，并论证了顽固度水平与$k$-确认规则之间的联系。我们表明，在每个攻击周期中，如果顽固程度高于$k$，对手就可以获得双重支出的免费机会。在每个周期中，对于给定的顽固度水平，我们严格制定双重消费的可能性有多大。我们进一步修改顽固政权中的攻击，以隐藏攻击并增加双重消费的概率。



## **44. Sparse Representations Improve Adversarial Robustness of Neural Network Classifiers**

稀疏表示提高神经网络分类器的对抗鲁棒性 cs.LG

Killian Steunou is the main contributor and corresponding author of  this work

**SubmitDate**: 2025-10-07    [abs](http://arxiv.org/abs/2509.21130v2) [paper-pdf](http://arxiv.org/pdf/2509.21130v2)

**Authors**: Killian Steunou, Théo Druilhe, Sigurd Saue

**Abstract**: Deep neural networks perform remarkably well on image classification tasks but remain vulnerable to carefully crafted adversarial perturbations. This work revisits linear dimensionality reduction as a simple, data-adapted defense. We empirically compare standard Principal Component Analysis (PCA) with its sparse variant (SPCA) as front-end feature extractors for downstream classifiers, and we complement these experiments with a theoretical analysis. On the theory side, we derive exact robustness certificates for linear heads applied to SPCA features: for both $\ell_\infty$ and $\ell_2$ threat models (binary and multiclass), the certified radius grows as the dual norms of $W^\top u$ shrink, where $W$ is the projection and $u$ the head weights. We further show that for general (non-linear) heads, sparsity reduces operator-norm bounds through a Lipschitz composition argument, predicting lower input sensitivity. Empirically, with a small non-linear network after the projection, SPCA consistently degrades more gracefully than PCA under strong white-box and black-box attacks while maintaining competitive clean accuracy. Taken together, the theory identifies the mechanism (sparser projections reduce adversarial leverage) and the experiments verify that this benefit persists beyond the linear setting. Our code is available at https://github.com/killian31/SPCARobustness.

摘要: 深度神经网络在图像分类任务中表现出色，但仍然容易受到精心设计的对抗性扰动的影响。这项工作重新审视了线性降维作为一种简单的、适应数据的防御。我们根据经验比较了标准主成分分析（PCA）与其稀疏变体（SPCA）作为下游分类器的前端特征提取器，并通过理论分析补充这些实验。在理论方面，我们为应用于SPCA特征的线性头部推导出精确的鲁棒性证书：对于$\ell_\infty$和$\ell_2 $威胁模型（二元和多类），认证半径随着$W &\top u$的双重规范缩小而增加，其中$W$是投影，$u$是头部重量。我们进一步表明，对于一般（非线性）头部，稀疏性通过Lipschitz合成论点减少了操作符规范界限，从而预测了较低的输入敏感性。从经验上看，投影后存在一个小型非线性网络，在强白盒和黑匣子攻击下，SPCA始终比PCA降级得更优雅，同时保持有竞争力的干净准确性。总而言之，该理论确定了机制（稀疏的预测减少了对抗杠杆），实验验证了这种好处在线性环境之外仍然存在。我们的代码可在https://github.com/killian31/SPCARobustness上获取。



## **45. SAFER: Advancing Safety Alignment via Efficient Ex-Ante Reasoning**

更安全：通过高效的前前推理推进安全一致 cs.CL

22 pages, 5 figures

**SubmitDate**: 2025-10-07    [abs](http://arxiv.org/abs/2504.02725v2) [paper-pdf](http://arxiv.org/pdf/2504.02725v2)

**Authors**: Kehua Feng, Keyan Ding, Yuhao Wang, Menghan Li, Fanjunduo Wei, Xinda Wang, Qiang Zhang, Huajun Chen

**Abstract**: Recent advancements in large language models (LLMs) have accelerated progress toward artificial general intelligence, yet their potential to generate harmful content poses critical safety challenges. Existing alignment methods often struggle to cover diverse safety scenarios and remain vulnerable to adversarial attacks. In this work, we propose SAFER, a framework for Safety Alignment via eFficient Ex-Ante Reasoning. Our approach instantiates structured Ex-Ante reasoning through initial assessment, rule verification, and path calibration, and embeds predefined safety rules to provide transparent and verifiable safety judgments. Specifically, our approach consists of two training stages: (1) supervised fine-tuning with synthetic traces to teach the multi-stage Ex-Ante reasoning, and (2) step-level reasoning preference optimization to jointly enhance safety, utility, and efficiency. Experiments on multiple open-source LLMs demonstrate that SAFER significantly enhances safety performance while maintaining helpfulness and response efficiency.

摘要: 大型语言模型（LLM）的最新进展加速了人工通用智能的发展，但它们生成有害内容的潜力带来了严峻的安全挑战。现有的对齐方法通常难以覆盖不同的安全场景，并且仍然容易受到对抗攻击。在这项工作中，我们提出了SAGER，这是一个通过eFficient Ex-Ante Reasoning进行安全调整的框架。我们的方法通过初始评估、规则验证和路径校准来实例化结构化的Ex-Ante推理，并嵌入预定义的安全规则以提供透明且可验证的安全判断。具体来说，我们的方法由两个训练阶段组成：（1）使用合成轨迹进行监督微调，以教授多阶段Ex-Ante推理，以及（2）分步推理偏好优化，以共同增强安全性、实用性和效率。对多个开源LLM的实验表明，SAGER显着增强了安全性能，同时保持了帮助性和响应效率。



## **46. DP-SNP-TIHMM: Differentially Private, Time-Inhomogeneous Hidden Markov Models for Synthesizing Genome-Wide Association Datasets**

DP-SNP-TIHM：用于合成全基因组关联数据集的差异私密、时间不均匀隐马尔科夫模型 cs.LG

**SubmitDate**: 2025-10-07    [abs](http://arxiv.org/abs/2510.05777v1) [paper-pdf](http://arxiv.org/pdf/2510.05777v1)

**Authors**: Shadi Rahimian, Mario Fritz

**Abstract**: Single nucleotide polymorphism (SNP) datasets are fundamental to genetic studies but pose significant privacy risks when shared. The correlation of SNPs with each other makes strong adversarial attacks such as masked-value reconstruction, kin, and membership inference attacks possible. Existing privacy-preserving approaches either apply differential privacy to statistical summaries of these datasets or offer complex methods that require post-processing and the usage of a publicly available dataset to suppress or selectively share SNPs.   In this study, we introduce an innovative framework for generating synthetic SNP sequence datasets using samples derived from time-inhomogeneous hidden Markov models (TIHMMs). To preserve the privacy of the training data, we ensure that each SNP sequence contributes only a bounded influence during training, enabling strong differential privacy guarantees. Crucially, by operating on full SNP sequences and bounding their gradient contributions, our method directly addresses the privacy risks introduced by their inherent correlations.   Through experiments conducted on the real-world 1000 Genomes dataset, we demonstrate the efficacy of our method using privacy budgets of $\varepsilon \in [1, 10]$ at $\delta=10^{-4}$. Notably, by allowing the transition models of the HMM to be dependent on the location in the sequence, we significantly enhance performance, enabling the synthetic datasets to closely replicate the statistical properties of non-private datasets. This framework facilitates the private sharing of genomic data while offering researchers exceptional flexibility and utility.

摘要: 单核苷酸多态性（SNP）数据集是遗传研究的基础，但在共享时会带来重大的隐私风险。SNP之间的相关性使得强大的对抗性攻击，如掩蔽值重建、亲属和成员推断攻击成为可能。现有的隐私保护方法要么将差异隐私应用于这些数据集的统计摘要，要么提供复杂的方法，这些方法需要后处理和使用公开可用的数据集来抑制或选择性地共享SNP。   在这项研究中，我们引入了一个创新的框架，用于生成合成SNP序列数据集，使用来自时间非齐次隐马尔可夫模型（TIHALGOT）的样本。为了保护训练数据的隐私，我们确保每个SNP序列在训练期间仅贡献有限的影响，从而实现强大的差异隐私保证。至关重要的是，通过对完整的SNP序列进行操作并限制其梯度贡献，我们的方法直接解决了其固有相关性带来的隐私风险。   通过在现实世界的1000个基因组数据集上进行的实验，我们使用隐私预算$\varepattack\in [1，10]$ at $\delta=10 '''' s来证明我们的方法的有效性。值得注意的是，通过允许Markov的过渡模型依赖于序列中的位置，我们显着增强了性能，使合成数据集能够紧密复制非私有数据集的统计属性。该框架促进了基因组数据的私人共享，同时为研究人员提供了卓越的灵活性和实用性。



## **47. Evidence of Cognitive Biases in Capture-the-Flag Cybersecurity Competitions**

捕获旗帜网络安全竞赛中认知偏见的证据 cs.CR

**SubmitDate**: 2025-10-07    [abs](http://arxiv.org/abs/2510.05771v1) [paper-pdf](http://arxiv.org/pdf/2510.05771v1)

**Authors**: Carolina Carreira, Anu Aggarwal, Alejandro Cuevas, Maria José Ferreira, Hanan Hibshi, Cleotilde Gonzalez

**Abstract**: Understanding how cognitive biases influence adversarial decision-making is essential for developing effective cyber defenses. Capture-the-Flag (CTF) competitions provide an ecologically valid testbed to study attacker behavior at scale, simulating real-world intrusion scenarios under pressure. We analyze over 500,000 submission logs from picoCTF, a large educational CTF platform, to identify behavioral signatures of cognitive biases with defensive implications. Focusing on availability bias and the sunk cost fallacy, we employ a mixed-methods approach combining qualitative coding, descriptive statistics, and generalized linear modeling. Our findings show that participants often submitted flags with correct content but incorrect formatting (availability bias), and persisted in attempting challenges despite repeated failures and declining success probabilities (sunk cost fallacy). These patterns reveal that biases naturally shape attacker behavior in adversarial contexts. Building on these insights, we outline a framework for bias-informed adaptive defenses that anticipate, rather than simply react to, adversarial actions.

摘要: 了解认知偏见如何影响对抗决策对于开发有效的网络防御至关重要。捕获旗帜（CTF）比赛提供了一个生态有效的测试平台，可以大规模研究攻击者的行为，模拟压力下的现实世界入侵场景。我们分析了来自大型教育CTF平台picoCTF的500，000多个提交日志，以识别具有防御意义的认知偏见的行为特征。我们专注于可用性偏差和沉没成本谬误，采用结合定性编码、描述性统计和广义线性建模的混合方法。我们的研究结果表明，参与者经常提交内容正确但格式不正确的标志（可用性偏差），并且尽管一再失败和成功概率下降（沉没成本谬误），但仍坚持尝试挑战。这些模式表明，偏见自然地塑造了敌对背景下的攻击者行为。在这些见解的基础上，我们概述了一个基于偏见的适应性防御框架，该框架可以预测而不是简单地对对抗行为做出反应。



## **48. Shortcuts Everywhere and Nowhere: Exploring Multi-Trigger Backdoor Attacks**

无处不在的捷径：探索多触发后门攻击 cs.LG

13 pages

**SubmitDate**: 2025-10-07    [abs](http://arxiv.org/abs/2401.15295v4) [paper-pdf](http://arxiv.org/pdf/2401.15295v4)

**Authors**: Yige Li, Jiabo He, Hanxun Huang, Jun Sun, Xingjun Ma, Yu-Gang Jiang

**Abstract**: Backdoor attacks have become a significant threat to the pre-training and deployment of deep neural networks (DNNs). Although numerous methods for detecting and mitigating backdoor attacks have been proposed, most rely on identifying and eliminating the ``shortcut" created by the backdoor, which links a specific source class to a target class. However, these approaches can be easily circumvented by designing multiple backdoor triggers that create shortcuts everywhere and therefore nowhere specific. In this study, we explore the concept of Multi-Trigger Backdoor Attacks (MTBAs), where multiple adversaries leverage different types of triggers to poison the same dataset. By proposing and investigating three types of multi-trigger attacks including \textit{parallel}, \textit{sequential}, and \textit{hybrid} attacks, we demonstrate that 1) multiple triggers can coexist, overwrite, or cross-activate one another, and 2) MTBAs easily break the prevalent shortcut assumption underlying most existing backdoor detection/removal methods, rendering them ineffective. Given the security risk posed by MTBAs, we have created a multi-trigger backdoor poisoning dataset to facilitate future research on detecting and mitigating these attacks, and we also discuss potential defense strategies against MTBAs. Our code is available at https://github.com/bboylyg/Multi-Trigger-Backdoor-Attacks.

摘要: 后门攻击已成为深度神经网络（DNN）预训练和部署的重大威胁。尽管已经提出了许多检测和减轻后门攻击的方法，但大多数方法依赖于识别和消除后门创建的“快捷方式”，该“快捷方式”将特定的源类链接到目标类。然而，通过设计多个后门触发器可以轻松规避这些方法，这些触发器可以在任何地方创建快捷方式，因此没有具体的地方。在这项研究中，我们探讨了多触发后门攻击（MTBA）的概念，即多个对手利用不同类型的触发器来毒害同一数据集。通过提出和研究三种类型的多触发器攻击，包括\textit{parallel}、\textit{serial}和\textit{hybrid}攻击，我们证明了1）多个触发器可以共存、覆盖或交叉激活彼此，2）MTA很容易打破大多数现有后门检测/删除方法背后的普遍捷径假设，使其无效。鉴于MTBA构成的安全风险，我们创建了一个多触发后门中毒数据集，以促进未来检测和减轻这些攻击的研究，我们还讨论了针对MTBA的潜在防御策略。我们的代码可在https://github.com/bboylyg/Multi-Trigger-Backdoor-Attacks上获取。



## **49. Geometry-Guided Adversarial Prompt Detection via Curvature and Local Intrinsic Dimension**

通过弯曲和局部固有维度的几何引导对抗提示检测 cs.CL

40 Pages, 6 figues

**SubmitDate**: 2025-10-07    [abs](http://arxiv.org/abs/2503.03502v2) [paper-pdf](http://arxiv.org/pdf/2503.03502v2)

**Authors**: Canaan Yung, Hanxun Huang, Christopher Leckie, Sarah Erfani

**Abstract**: Adversarial prompts are capable of jailbreaking frontier large language models (LLMs) and inducing undesirable behaviours, posing a significant obstacle to their safe deployment. Current mitigation strategies primarily rely on activating built-in defence mechanisms or fine-tuning LLMs, both of which are computationally expensive and can sacrifice model utility. In contrast, detection-based approaches are more efficient and practical for deployment in real-world applications. However, the fundamental distinctions between adversarial and benign prompts remain poorly understood. In this work, we introduce CurvaLID, a novel defence framework that efficiently detects adversarial prompts by leveraging their geometric properties. It is agnostic to the type of LLM, offering a unified detection framework across diverse adversarial prompts and LLM architectures. CurvaLID builds on the geometric analysis of text prompts to uncover their underlying differences. We theoretically extend the concept of curvature via the Whewell equation into an $n$-dimensional word embedding space, enabling us to quantify local geometric properties, including semantic shifts and curvature in the underlying manifolds. To further enhance our solution, we leverage Local Intrinsic Dimensionality (LID) to capture complementary geometric features of text prompts within adversarial subspaces. Our findings show that adversarial prompts exhibit distinct geometric signatures from benign prompts, enabling CurvaLID to achieve near-perfect classification and outperform state-of-the-art detectors in adversarial prompt detection. CurvaLID provides a reliable and efficient safeguard against malicious queries as a model-agnostic method that generalises across multiple LLMs and attack families.

摘要: 对抗性提示能够越狱前沿大型语言模型（LLM）并引发不良行为，对其安全部署构成重大障碍。当前的缓解策略主要依赖于激活内置防御机制或微调LLM，这两者计算成本很高，并且可能会牺牲模型效用。相比之下，基于检测的方法对于在现实世界应用程序中部署更有效和实用。然而，对抗性提示和良性提示之间的根本区别仍然知之甚少。在这项工作中，我们引入了CurvaLID，这是一种新型防御框架，可以通过利用其几何属性来有效检测对抗提示。它与LLM类型无关，提供跨不同对抗提示和LLM架构的统一检测框架。CurvaLID基于文本提示的几何分析，以揭示其潜在差异。从理论上讲，我们通过Whewell方程将弯曲的概念扩展到$n维单词嵌入空间，使我们能够量化局部几何属性，包括底层流中的语义移动和弯曲。为了进一步增强我们的解决方案，我们利用局部本质模糊性（LID）来捕获对抗子空间中文本提示的补充几何特征。我们的研究结果表明，对抗性提示表现出与良性提示不同的几何特征，使CurvaLID能够实现近乎完美的分类，并在对抗性提示检测方面优于最先进的检测器。CurvaLID作为一种模型不可知的方法，可在多个LLM和攻击系列中推广，提供可靠且高效的防范恶意查询的保护措施。



## **50. Benchmarking the Robustness of Agentic Systems to Adversarially-Induced Harms**

对平衡系统的稳健性进行基准测试以应对不利引起的伤害 cs.LG

54 Pages

**SubmitDate**: 2025-10-07    [abs](http://arxiv.org/abs/2508.16481v2) [paper-pdf](http://arxiv.org/pdf/2508.16481v2)

**Authors**: Jonathan Nöther, Adish Singla, Goran Radanovic

**Abstract**: Ensuring the safe use of agentic systems requires a thorough understanding of the range of malicious behaviors these systems may exhibit when under attack. In this paper, we evaluate the robustness of LLM-based agentic systems against attacks that aim to elicit harmful actions from agents. To this end, we propose a novel taxonomy of harms for agentic systems and a novel benchmark, BAD-ACTS, for studying the security of agentic systems with respect to a wide range of harmful actions. BAD-ACTS consists of 4 implementations of agentic systems in distinct application environments, as well as a dataset of 188 high-quality examples of harmful actions. This enables a comprehensive study of the robustness of agentic systems across a wide range of categories of harmful behaviors, available tools, and inter-agent communication structures. Using this benchmark, we analyze the robustness of agentic systems against an attacker that controls one of the agents in the system and aims to manipulate other agents to execute a harmful target action. Our results show that the attack has a high success rate, demonstrating that even a single adversarial agent within the system can have a significant impact on the security. This attack remains effective even when agents use a simple prompting-based defense strategy. However, we additionally propose a more effective defense based on message monitoring. We believe that this benchmark provides a diverse testbed for the security research of agentic systems. The benchmark can be found at github.com/JNoether/BAD-ACTS

摘要: 确保代理系统的安全使用需要彻底了解这些系统在受到攻击时可能表现出的恶意行为范围。在本文中，我们评估了基于LLM的代理系统针对旨在引发代理有害行为的攻击的稳健性。为此，我们提出了一种新型的代理系统危害分类法和一种新型基准BAD-SYS，用于研究代理系统针对广泛有害行为的安全性。BAD-SYS由不同应用环境中的4个代理系统实现以及包含188个有害行为高质量示例的数据集组成。这使得能够对各种有害行为、可用工具和代理间通信结构的代理系统的稳健性进行全面研究。使用这个基准测试，我们分析的鲁棒性的代理系统对攻击者，控制系统中的代理之一，目的是操纵其他代理执行有害的目标行动。我们的研究结果表明，攻击具有很高的成功率，表明即使是系统中的单个对抗代理也会对安全性产生重大影响。即使代理使用简单的基于预算的防御策略，此攻击仍然有效。然而，我们还提出了一种基于消息监控的更有效的防御。我们相信该基准为代理系统的安全研究提供了多样化的测试平台。基准测试可以在github.com/JNoether/BAD-ACTS上找到



