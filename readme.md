This is the Google Hashcode 2020 problem.

https://storage.googleapis.com/coding-competitions.appspot.com/HC/2020/hashcode2020_final_round.pdf

In essence: we are given a WxH grid with defined mountpoints.

Tasks consisting of 1 or more assembly points are defined. Some tasks have higher reward points than others.

We get to choose which mountpoints to place R robot arms which will carry out the tasks.

Each step of time, each arm can move Up/Down/L/R or wait. Arms cannot occupy cells which are occupied or contain mountpoint.

here's my strategy:

1. compute the path and distance from every mountpoint to every first assembly point of every task.
2. assign each task to its nearest mountpoint, then assign an arm to each mountpoint.
3. now loop for each time step:

   - move each arm according to its assigned route to the assembly point
   - check for collision
   - if collision, find new route

at this point, things get complicated :sweat_smile:

but this is a good milestone to reach, before we start tackling the complicated stuff. at this milestone we should have github ready, we'd be familiar with the code & each other's coding style.
