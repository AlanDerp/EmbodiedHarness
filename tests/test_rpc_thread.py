import queue
import threading
import unittest

from embodied_harness.rpc import RpcClient, SerialRpcServer


class NativeThreadTests(unittest.TestCase):
    def test_requests_keep_simulator_owner_thread(self):
        ready = queue.Queue()
        failures = queue.Queue()

        def native_worker():
            owner = threading.get_ident()
            def step():
                self.assertEqual(threading.get_ident(), owner)
                return owner
            try:
                with SerialRpcServer(('127.0.0.1', 0), 'test', {'step': step}) as server:
                    ready.put(server)
                    server.serve_forever(poll_interval=.01)
            except BaseException as error:
                failures.put(error)

        thread = threading.Thread(target=native_worker, daemon=True)
        thread.start()
        server = ready.get(timeout=5)
        try:
            client = RpcClient('http://127.0.0.1:' + str(server.server_port), 'test', timeout_s=5)
            for _ in range(3):
                self.assertEqual(client.call('step'), thread.ident)
        finally:
            server.shutdown()
            thread.join(timeout=5)
        self.assertFalse(thread.is_alive())
        if not failures.empty():
            raise failures.get()
