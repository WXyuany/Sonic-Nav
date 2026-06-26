#!/usr/bin/env -S /usr/bin/python3
import os,sys,math,time,rclpy,msgpack
os.environ.update({'RMW_IMPLEMENTATION':'rmw_fastrtps_cpp','ROS_LOCALHOST_ONLY':'1','ROS_DOMAIN_ID':'42'})
from rclpy.node import Node
from geometry_msgs.msg import PoseStamped
from std_msgs.msg import ByteMultiArray

class GoalFollower(Node):
    def __init__(self):
        super().__init__('goal_follower')
        self.pub = self.create_publisher(ByteMultiArray,'ControlPolicy/upper_body_pose',10)
        self.create_subscription(PoseStamped,'/goal_pose',self.on_goal,10)
        self.goal = None
        self.last_vx = 0.0
        self.last_vy = 0.0
        self.last_vw = 0.0
        self.started = False
        self.timer = self.create_timer(0.1,self.tick)
        self.get_logger().info('Goal follower ready. Set 2D Goal Pose in RViz.')

    def on_goal(self,msg):
        self.goal = (msg.pose.position.x, msg.pose.position.y)
        self.get_logger().info(f'Goal: ({self.goal[0]:.2f}, {self.goal[1]:.2f})')

    def tick(self):
        pl = {'toggle_policy_action': not self.started,'locomotion_mode':0,'base_height_command':0.78}
        self.started = True
        if self.goal is None:
            pl['navigate_cmd'] = [0,0,0]
        else:
            gx,gy = self.goal
            dist = math.hypot(gx,gy)
            if dist < 0.3:
                self.goal = None
                pl['navigate_cmd'] = [0,0,0]
                self.get_logger().info('Goal reached!')
            else:
                angle = math.atan2(gy,gx)
                self.last_vx = min(0.5, dist*0.3)
                self.last_vw = angle * 0.5
                pl['navigate_cmd'] = [self.last_vx, 0, self.last_vw]
        m = ByteMultiArray()
        m.data = [bytes([b]) for b in msgpack.packb(pl,use_bin_type=True)]
        self.pub.publish(m)

def main():
    rclpy.init()
    n = GoalFollower()
    try: rclpy.spin(n)
    except KeyboardInterrupt: pass
    n.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
