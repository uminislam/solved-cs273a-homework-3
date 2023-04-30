Download Link: https://assignmentchef.com/product/solved-cs273a-homework-3
<br>
<h2>Problem 1: Logistic Regression</h2>

<table width="117">

 <tbody>

  <tr>

   <td width="117">logisticClassify2</td>

  </tr>

 </tbody>

</table>

In this problem, we’ll build a logistic regression classifier and train it on separable and non-separable data. Since it will be specialized to binary classification, we’ve named the class.

We’ll start by building two binary classification problems, one separable and the other not:

<table width="624">

 <tbody>

  <tr>

   <td width="624">iris = np.genfromtxt(“data/iris.txt”,delimiter=None)X, Y = iris[:,0:2], iris[:,-1] # get first two features &amp; targetX,Y = ml.shuffleData(X,Y)           # reorder randomly (important later) X,_ = rescale(X)               # works much better on rescaled dataXA, YA = X[Y&lt;2,:], Y[Y&lt;2]                                             # get class 0 vs 1XB, YB = X[Y&gt;0,:], Y[Y&gt;0]                                             # get class 1 vs 2</td>

  </tr>

 </tbody>

</table>

1

2

3

4

5

6

7

For this problem, we are focused on the learning algorithm, rather than performance — so, we will not bother creating training and validation splits; just use all your data for training.

<table width="50">

 <tbody>

  <tr>

   <td width="50">permute</td>

  </tr>

 </tbody>

</table>

<strong>Note: </strong>The code uses numpy’sto iterate over data randomly; should avoid issues due to the default order of the data (by class). Similarly, rescaling and centering the data may help speed up convergence as well.

<ol>

 <li>Show the two classes in a scatter plot (one for each data set) and verify that one data set is linearly separable while the other is not. <em>(5 points)</em></li>

</ol>

<table width="138">

 <tbody>

  <tr>

   <td width="138">logisticClassify2.py</td>

  </tr>

 </tbody>

</table>

<table width="84">

 <tbody>

  <tr>

   <td width="84">plotBoundary</td>

  </tr>

 </tbody>

</table>

<ol start="2">

 <li>Write (fill in) the functioninto compute the points on the decision boundary. In particular, you only need to make sure x2b is set correctly using theta . This will plot the data &amp; boundary quickly, which is useful for visualizing the model during training. To demo your function plot the decision boundary corresponding to the classifier</li>

</ol>

sign( .5 <em>− </em>.25<em>x</em><sub>1 </sub>+ 1<em>x</em><sub>2 </sub>)

along with the A data, and again with the B data; these fixed parameters will look like an OK classifier on one data set, but a poor classifier on the other.

You can create a “blank” learner and set the weights by:

<table width="591">

 <tbody>

  <tr>

   <td width="276"><strong>import </strong>mltools as ml <strong>from </strong>logisticClassify2 <strong>import </strong>* learner = logisticClassify2();</td>

   <td width="315"># create “blank” learner</td>

  </tr>

  <tr>

   <td width="276">learner.classes = np.unique(YA)</td>

   <td width="315"># define class labels using YA or YB</td>

  </tr>

 </tbody>

</table>

1

2

3

4

5

<table width="591">

 <tbody>

  <tr>

   <td width="591">wts = np.array([theta0,theta1,theta2]); # TODO: fill in valueslearner.theta = wts;                                                                      # set the learner’s parameters</td>

  </tr>

 </tbody>

</table>

6

7

Include the lines of code you added to the function, and the two generated plots. <em>(10 points)</em>

<table width="171">

 <tbody>

  <tr>

   <td width="171">logisticClassify2.predict</td>

  </tr>

 </tbody>

</table>

<ol start="3">

 <li>Complete thefunction to make predictions for your classifier. Verify that your function works by computing &amp; reporting the error rate of the classifier in the previous part on both data sets A and B. (The error rate on one should be <em>≈ </em>0505, and higher on the other.) Note that, in the code, the two classes are stored in the variable self.classes , with the first entry being the “negative” class (or class 0), and the second entry being the “positive” class, so you want to have different learner objects for each dataset, and you use learner.err directly.</li>

</ol>

Include the function definition and the two computed errors. <em>(10 points)</em>

<table width="97">

 <tbody>

  <tr>

   <td width="97">plotClassify2D</td>

  </tr>

 </tbody>

</table>

<table width="50">

 <tbody>

  <tr>

   <td width="50">predict</td>

  </tr>

 </tbody>

</table>

<ol start="4">

 <li>Verify that your predict code matches your boundary plot by usingwith your manually constructed learner on the two data sets. This will callon a dense grid of points, and you should find that the resulting decision boundary matches the one you computed analytically. <em>(5 points)</em></li>

 <li>In the provided code, we first transform the classes in the data <em>Y </em>into <em>YY </em>, with canonical labels for the two classes: “class 0” (negative) and “class 1” (positive). In our notation, let <em>r</em><sup>(<em>j</em>) </sup>= <em>x</em><sup>(<em>j</em>) </sup><em>θ <sup>T </sup></em>be the linear response of the classifier, and <em>σ </em>is the standard logistic function</li>

</ol>

<em>σ</em>(<em>r</em>)= 1 + exp(<em>−r</em>)<em><sup>−</sup></em><sup>1</sup>.

The logistic negative log likelihood loss for a single data point <em>j </em>is then

<em>J<sub>j</sub></em>(<em>θ</em>)= <em>−y</em><sup>(<em>j</em>) </sup>log <em>σ</em>(<em>x</em><sup>(<em>j</em>)</sup><em>θ <sup>T </sup></em>) <em>− </em>(1 <em>− y</em><sup>(<em>j</em>)</sup>) log(1 <em>−σ</em>(<em>x</em><sup>(<em>j</em>)</sup><em>θ <sup>T </sup></em>))

where <em>y</em><sup>(<em>j</em>) </sup>is either 0 or 1. Derive the gradient of the negative log likelihood <em>J<sub>j </sub></em>for logistic regression, and

give it in your report. (You will need this in your gradient descent code for the next part.)

Provide the gradient equations for <em><sub>∂θ</sub><u><sup>∂</sup></u></em>0 <em>J<sub>j</sub></em>, <em><sub>∂θ</sub><u><sup>∂</sup></u></em>1 <em>J<sub>j</sub></em>, and       <em>(10 points)</em>

<ol start="6">

 <li>Complete train function to perform stochastic gradient descent on the logistic loss function. This will require that you fill in:

  <ul>

   <li>computing the surrogate loss function at each epoch (<em>J </em>= <em><sub>m</sub></em><u><sup>1 </sup></u><sup>P</sup><em>J<sub>j</sub></em>, from the previous part);</li>

   <li>computing the response (<em>r</em><sup>(<em>j</em>) </sup>and gradient associated with each data point <em>x</em><sup>(<em>j</em>)</sup>, <em>y</em><sup>(<em>j</em>)</sup>;</li>

  </ul></li>

</ol>

<table width="70">

 <tbody>

  <tr>

   <td width="70">stopEpochs</td>

  </tr>

 </tbody>

</table>

<table width="50">

 <tbody>

  <tr>

   <td width="50">stopTol</td>

  </tr>

 </tbody>

</table>

<ul>

 <li>a stopping criterion consisting of two conditions (stop when either you have reachedepochs or <em>J </em>has not changed by more thansince the last epoch). Include the complete implementation of train . <em>(25 points)</em></li>

</ul>

<ol start="7">

 <li>Run train for your logistic regression classifier on both data sets (A and B). Describe your parameter choices for each dataset (stepsize, etc.) and include plots showing the convergence of the surrogate loss and error rate (e.g., the loss values as a function of epoch during gradient descent), and the final converged classifier with the data (the included train function does that for you already). <em>(10 points)</em></li>

</ol>

<table width="222">

 <tbody>

  <tr>

   <td width="77">raw_input()</td>

   <td width="95">(Python 2.7) or</td>

   <td width="50">input()</td>

  </tr>

 </tbody>

</table>

<strong>Note: </strong>Debugging machine learning algorithms can be quite challenging, since the results of the algorithm are highly data-dependent, and often somewhat randomized (initialization, etc.). I suggest starting with an extremely small step size and verifying both that the learner’s prediction evolves slowly in the correct direction, and that the objective function <em>J </em>decreases monotonically. If that works, go to larger step sizes to observe the behavior. I often manually step through the code — for example by pausing after each parameter update using(Python 3) – so that I can examine its behavior. You can also (of course) use a more sophisticated debugger.

<table width="90">

 <tbody>

  <tr>

   <td width="90">pyplot.draw()</td>

  </tr>

 </tbody>

</table>

<strong>Note on plotting: </strong>The code generates plots as the algorithm runs, so you can see its behavior over time; this is done with. Run your code either interactively or as a script to see these display over time; refer to discussion notebooks on how to plot the loss over time in Jupyter.

<ol start="8">

 <li><strong>Extra Credit </strong><em>(10 points)</em>: Add an L2 regularization term (+<em>α</em><sup>P</sup><em><sub>i </sub>θ<sub>i</sub></em><sup>2</sup>) to your surrogate loss function, and update the gradient and your code to reflect this addition. Try re-running your learner with some regularization (e.g. <em>α</em>= 2) and see how different the resulting parameters are. Find a value of <em>α </em>that gives noticeably different results &amp; explain them.</li>

</ol>

(a)                                           (b)                                           (c)                                           (d)

Figure 1: Four datasets to test whether they can be <em>shattered </em>by a given classifier, i.e. can the classifier exactly separate their all possible binary colorings. <strong>No three data points are on a line.</strong>

<h2>Problem 2: Shattering and VC Dimension</h2>

Consider the data points in Figure 1 which have two real-valued features <em>x</em><sub>1</sub>, <em>x</em><sub>2</sub>. We are also giving a few learners below. For the learners below, <em>T</em>[<em>z</em>] is the sign threshold function, <em>T</em>[<em>z</em>]=+1 for <em>z ≥ </em>0 and <em>T</em>[<em>z</em>]= <em>−</em>1 for <em>z &lt; </em>0.

The learner parameters <em>a</em>, <em>b</em>, <em>c</em>, . . . are real-valued scalars, and each data point has two real-valued features <em>x</em><sub>1</sub>, <em>x</em><sub>2</sub>.

Which of the four datasets can be shattered by each learner? Give a brief explanation/justification and use your results to guess the VC dimension of the classifier (you do not have to give a formal proof, just your reasoning).

<ol>

 <li><em>T</em>(<em>a </em>+ <em>bx</em><sub>1 </sub>) <em>(5 points)</em></li>

 <li><em>T</em>((<em>a ∗ b</em>)<em>x</em><sub>1 </sub>+(<em>c/a</em>)<em>x</em><sub>2 </sub>) <em>(5 points)</em></li>

 <li><em>T</em>((<em>x</em><sub>1 </sub><em>− a</em>)<sup>2 </sup>+(<em>x</em><sub>2 </sub><em>− b</em>)<sup>2 </sup>+ <em>c </em>) <em>(5 points)</em></li>

 <li><em>(5 points)</em></li>

</ol>

<h2>Statement of Collaboration</h2>

It is <strong>mandatory </strong>to include a <em>Statement of Collaboration </em>in each submission, with respect to the guidelines below. Include the names of everyone involved in the discussions (especially in-person ones), and what was discussed.

All students are required to follow the academic honesty guidelines posted on the course website. For programming assignments, in particular, I encourage the students to organize (perhaps using Campuswire) to discuss the task descriptions, requirements, bugs in my code, and the relevant technical content <em>before </em>they start working on it. However, you should not discuss the specific solutions, and, as a guiding principle, you are not allowed to take anything written or drawn away from these discussions (i.e. no photographs of the blackboard,

written notes, referring to Campuswire, etc.). Especially <em>after </em>you have started working on the assignment, try to restrict the discussion to Campuswire as much as possible, so that there is no doubt as to the extent of your collaboration.