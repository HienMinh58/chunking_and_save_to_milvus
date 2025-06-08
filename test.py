from pymilvus import connections
try:
    connections.connect(host='localhost', port='19530')
    print("Kết nối thành công với localhost")
except Exception as e:
    print(f"Lỗi localhost: {e}")
    try:
        connections.connect(host='host.docker.internal', port='19530')
        print("Kết nối thành công với host.docker.internal")
    except Exception as f:
        print(f"Lỗi host.docker.internal: {f}")