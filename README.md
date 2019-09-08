# moseq2-app
MoSeq2 Web Application Platform used to run all of the MoSeq2 tools in a GUI.
***

## Install and Run Steps
 * Clone this directory to your local machine.
 * Replace `{username}:{password}` with your github username and password for private repo authentication in `requirements.prod.txt`.
 * Install `docker`. Follow the following link for your corresponding operating system:
    * [Windows 10](https://runnable.com/docker/install-docker-on-windows-10)
    * [MacOS](https://docs.docker.com/docker-for-mac/install/)
    * [Ubuntu](https://docs.docker.com/install/linux/docker-ce/ubuntu/)
    * [Debian](https://docs.docker.com/install/linux/docker-ce/debian/)
    * [CentOS](https://docs.docker.com/install/linux/docker-ce/centos/)
 * Once Docker is installed, run the bash command: ```./INSTALL```
    * Enter your github username and password as prompted in order to clone the private repositories.
    * Wait for the install to finish.
 * It will create and build your Docker images and container for **`moseq2-app`**.
    * It will also run the app in the url: `0.0.0.0:4000/`
 
