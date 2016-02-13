import Queue
import threading
from .. import utils


class ThreadedGenerator(object):

    def __init__(self, generator, buffer_size=2):
        """
        Generator that runs a slow source generator in a separate thread.
        Beware of the GIL!
        buffer_size: the maximal number of items to pre-generate
                     (length of the buffer)

        """
        if buffer_size < 2:
            raise RuntimeError("Minimal buffer size is 2!")
        self.generator = generator
        self.buffer_size = buffer_size
        # the effective buffer size is one less, because the generation process
        # will generate one extra element and block until there is room in the
        # buffer.
        self.queue_ = Queue.Queue(maxsize=self.buffer_size - 1)

    def open(self):

        def _buffered_generation_thread(source_gen, queue):
            try:
                for data in source_gen:
                    queue.put(data, block=True)
            except Exception as e:
                utils.exception("Exception on separate thread: %s" % e)
                queue.put(e, block=True)
            except KeyboardInterrupt:
                utils.exception("KeyboardInterrupt on separate thread")
                queue.put(Exception("KeyboardInterrupt on separate thread"),
                          block=True)
            # sentinel: signal the end of the iterator
            # unfortunately this does not suffice as a signal: if queue.get()
            # was called and subsequently the queue is closed, it will block
            # forever
            queue.put(None)
            # this seems to break?
            # queue.close()

        self.thread = threading.Thread(
            target=_buffered_generation_thread,
            args=(self.generator,
                  self.queue_))
        self.thread.daemon = True
        self.thread.start()

        def gen_from_queue():
            for data in iter(self.queue_.get, None):
                if isinstance(data, Exception):
                    raise data
                else:
                    yield data

        return gen_from_queue()

    def close(self):
        # TODO clean up / stop thread
        # http://stackoverflow.com/questions/323972/is-there-any-way-to-kill-a-thread-in-python
        pass

    def __enter__(self):
        return self.open()

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()
        # don't supress any exception
        return False
