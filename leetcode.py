class ListNode:
	def __init__(self, val=0, next=None):
		self.val = val
		self.next = next


class Solution:
	def getLengthOf(self, l: ListNode):
		length = 1
		while l.next is not None:
			length += 1
		return length

	def addTwoNumbers(self, l1: ListNode, l2: ListNode):
		if l1.val == 0 and l1.next == None:
			return l2
		if l2.val == 0 and l2.next == None:
			return l1
		l3 = l4 = ListNode()
		len1 = self.getLengthOf(l1)
		len2 = self.getLengthOf(l2)
		length = min(len1, len2)
		i = 0
		j = 1
		while i <= length:
			l = l1.val + l2.val
			if l >= 10:
				l4.val = l - 10
				j = 1
			else:
				l4.val = l
				j = 0
				i += 1
